#!/usr/bin/env python3
"""Install the RAG index template and create the first versioned index safely."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.parse

from http_utils import (
    ServiceError,
    connection_from_environment,
    request,
    request_json_with,
    parse_bool,
    validate_identifier,
)
from index_contract import (
    IndexContract,
    alias_indexes,
    load_template,
    mapping_for_index,
    render_template,
    validate_mapping,
)


DEFAULT_TEMPLATE = pathlib.Path(__file__).resolve().parents[1] / "config" / "index-template.json"


def contract_from_environment() -> IndexContract:
    return IndexContract(
        schema_version=os.getenv("RAG_SCHEMA_VERSION", "1"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "SET_ME"),
        embedding_revision=os.getenv("RAG_EMBEDDING_REVISION", "SET_ME"),
        embedding_dimension=int(os.getenv("RAG_EMBEDDING_DIMENSION", "1024")),
        query_instruction=os.getenv("RAG_QUERY_INSTRUCTION", ""),
        document_instruction=os.getenv("RAG_DOCUMENT_INSTRUCTION", ""),
        vectors_normalized=parse_bool(
            os.getenv("RAG_VECTORS_NORMALIZED", "true"), label="RAG_VECTORS_NORMALIZED"
        ),
        tokenizer_revision=os.getenv("RAG_TOKENIZER_REVISION", "SET_ME"),
        max_input_tokens=int(os.getenv("RAG_MAX_INPUT_TOKENS", "8192")),
        parser_version=os.getenv("RAG_PARSER_VERSION", "1"),
        chunker_version=os.getenv("RAG_CHUNKER_VERSION", "1"),
    )


def encoded(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-file", type=pathlib.Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--template-name", default=os.getenv("RAG_INDEX_TEMPLATE_NAME", "puncture-rag-template-v1")
    )
    parser.add_argument("--index", default=os.getenv("RAG_INDEX_NAME", "puncture-rag-v000001"))
    parser.add_argument("--read-alias", default=os.getenv("RAG_READ_ALIAS", "puncture-rag-read"))
    parser.add_argument("--write-alias", default=os.getenv("RAG_WRITE_ALIAS", "puncture-rag-write"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        template_name = validate_identifier(args.template_name, label="template name")
        index_name = validate_identifier(args.index, label="index", versioned_index=True)
        read_alias = validate_identifier(args.read_alias, label="read alias")
        write_alias = validate_identifier(args.write_alias, label="write alias")
        if len({index_name, read_alias, write_alias}) != 3:
            raise ValueError("index and alias names must be distinct")
        contract = contract_from_environment()
        rendered = render_template(load_template(args.template_file), contract)
        patterns = rendered.get("index_patterns")
        if not isinstance(patterns, list) or not any(
            isinstance(pattern, str) and index_name.startswith(pattern.removesuffix("*"))
            for pattern in patterns
            if pattern.endswith("*")
        ):
            raise ValueError("versioned index name is not covered by the template pattern")
        if args.dry_run:
            print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        config = connection_from_environment()
        alias_state: dict[str, tuple[str, ...]] = {}
        for alias in (read_alias, write_alias):
            status, response = request_json_with(
                config,
                "GET",
                f"/_alias/{encoded(alias)}",
                allowed_statuses=(200, 404),
            )
            alias_state[alias] = () if status == 404 else alias_indexes(response, alias)
        existing = set(alias_state[read_alias]) | set(alias_state[write_alias])
        if existing and alias_state != {read_alias: (index_name,), write_alias: (index_name,)}:
            raise ServiceError("bootstrap refuses to replace an existing live alias")

        request_json_with(
            config,
            "PUT",
            f"/_index_template/{encoded(template_name)}",
            payload=rendered,
            allowed_statuses=(200,),
        )

        existence = request(
            "HEAD",
            config.endpoint,
            f"/{encoded(index_name)}",
            username=config.username,
            password=config.password,
            timeout=config.timeout,
            ca_file=config.ca_file,
            insecure=config.insecure,
            allowed_statuses=(200, 404),
        )
        created = existence.status == 404
        if created:
            request_json_with(
                config,
                "PUT",
                f"/{encoded(index_name)}",
                payload={},
                allowed_statuses=(200,),
            )

        _, mapping_response = request_json_with(
            config, "GET", f"/{encoded(index_name)}/_mapping", allowed_statuses=(200,)
        )
        validate_mapping(mapping_for_index(mapping_response, index_name), contract)

        if not existing:
            request_json_with(
                config,
                "POST",
                "/_aliases",
                payload={
                    "actions": [
                        {"add": {"index": index_name, "alias": read_alias}},
                        {
                            "add": {
                                "index": index_name,
                                "alias": write_alias,
                                "is_write_index": True,
                            }
                        },
                    ]
                },
                allowed_statuses=(200,),
            )
        print(
            json.dumps(
                {
                    "status": "READY",
                    "template": template_name,
                    "index": index_name,
                    "created": created,
                    "read_alias": read_alias,
                    "write_alias": write_alias,
                    "embedding_dimension": contract.embedding_dimension,
                    "vectors_normalized": contract.vectors_normalized,
                    "tokenizer_revision": contract.tokenizer_revision,
                    "max_input_tokens": contract.max_input_tokens,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ServiceError, ValueError, OSError) as exc:
        print(f"BOOTSTRAP_FAILED {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
