"""Offline ingestion, lifecycle, and atomic-index tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puncture_agent.rag import (  # noqa: E402
    DeterministicEmbeddingBackend,
    HeadingAwareChunker,
    InMemoryHybridIndex,
    MarkdownSectionParser,
    RagIngestionService,
    RagServiceError,
    SourceDocument,
)


def source(**overrides) -> SourceDocument:
    values = {
        "document_id": "planning-rules",
        "title": "Planning Rules",
        "source_uri": "internal://knowledge/planning-rules",
        "source_type": "markdown",
        "module": "path_planning",
        "version": "v3",
        "status": "active",
        "owner": "algorithm-team",
        "access_scopes": ("algorithm-team",),
        "content": "# Safety\nERR_PATH_42 requires a 5 mm clearance around the complete needle path.",
        "updated_at": "2026-07-01T00:00:00Z",
        "metadata": {"category": "approved", "language": "en"},
    }
    values.update(overrides)
    return SourceDocument(**values)


class FailingEmbedding(DeterministicEmbeddingBackend):
    def embed_documents(self, texts):
        if any("FAIL_EMBED" in text for text in texts):
            raise RuntimeError("scripted embedding outage")
        return super().embed_documents(texts)


class LimitedBatchEmbedding(DeterministicEmbeddingBackend):
    def __init__(self):
        super().__init__("embedding-test", "rev-1", 32)
        self.batch_sizes = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        if len(texts) > 2:
            raise RuntimeError("batch too large")
        return super().embed_documents(texts)


class DifferentInstructionEmbedding(DeterministicEmbeddingBackend):
    @property
    def query_instruction(self) -> str:
        return "A different query instruction."


class TinyContextEmbedding(DeterministicEmbeddingBackend):
    @property
    def max_input_tokens(self) -> int:
        return 8


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = InMemoryHybridIndex("test-rag")
        self.embedding = DeterministicEmbeddingBackend("embedding-test", "rev-1", 32)
        self.service = RagIngestionService(self.backend, self.embedding)

    def test_heading_hierarchy_parent_link_and_stable_chunk_ids(self) -> None:
        document = source(
            content="# API\nIntro.\n## Errors\nERR_PATH_42 is retryable.\n### Signature\n```cpp\nResult plan(const Case& c);\n```"
        )
        first = self.service.prepare(document)
        second = self.service.prepare(document)

        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunks],
            [chunk.chunk_id for chunk in second.chunks],
        )
        self.assertTrue(any(chunk.section_path[-2:] == ("API", "Errors") for chunk in first.chunks))
        parent_ids = {parent.parent_id for parent in first.parents}
        self.assertTrue(all(chunk.parent_id in parent_ids for chunk in first.chunks))

    def test_table_and_api_block_remain_coherent(self) -> None:
        document = source(
            content=(
                "# Error Contract\n"
                "| Code | Retryable | Meaning |\n"
                "| --- | --- | --- |\n"
                + "\n".join(f"| ERR_{index} | no | Meaning {index} |" for index in range(20))
                + "\n\n```cpp\nResult validate_label_schema(const Input& input);\n```"
            )
        )
        parser = MarkdownSectionParser()
        chunker = HeadingAwareChunker(target_tokens=16, overlap_tokens=4)
        _, chunks = chunker.build_records(document, parser.parse(document))

        table_chunks = [chunk.text for chunk in chunks if "| Code |" in chunk.text]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn("ERR_19", table_chunks[0])
        code_chunks = [chunk.text for chunk in chunks if "validate_label_schema" in chunk.text]
        self.assertEqual(len(code_chunks), 1)
        self.assertTrue(code_chunks[0].strip().startswith("```cpp"))

    def test_repeated_page_header_and_footer_are_removed(self) -> None:
        document = source(
            content=(
                "CONFIDENTIAL\n# One\nFirst rule.\nPage 1\f"
                "CONFIDENTIAL\n# Two\nSecond rule.\nPage 1"
            )
        )
        sections = MarkdownSectionParser().parse(document)
        joined = "\n".join(section.text for section in sections)
        self.assertNotIn("CONFIDENTIAL", joined)
        self.assertNotIn("Page 1", joined)
        self.assertIn("First rule", joined)
        self.assertIn("Second rule", joined)

    def test_acl_module_version_status_and_versions_are_inherited(self) -> None:
        prepared = self.service.prepare(source())
        for chunk in prepared.chunks:
            self.assertEqual(chunk.access_scopes, ("algorithm-team",))
            self.assertEqual(chunk.module, "path_planning")
            self.assertEqual(chunk.version, "v3")
            self.assertEqual(chunk.status, "active")
            self.assertEqual(chunk.parser_version, "markdown-normalizer-v1")
            self.assertEqual(chunk.chunker_version, "heading-parent-child-v1")

    def test_chinese_without_whitespace_obeys_chunk_budget(self) -> None:
        document = source(content="# 规则\n" + "路径安全距离" * 200)
        parser = MarkdownSectionParser()
        chunker = HeadingAwareChunker(target_tokens=40, overlap_tokens=8)
        _, chunks = chunker.build_records(document, parser.parse(document))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 40 for chunk in chunks))

    def test_multi_block_chinese_overlap_stays_within_budget(self) -> None:
        document = source(content="# 规则\n" + "十个汉字正好一段" * 2 + "\n\n" + "第二段也是十个字" * 2)
        parser = MarkdownSectionParser()
        chunker = HeadingAwareChunker(target_tokens=16, overlap_tokens=8)
        _, chunks = chunker.build_records(document, parser.parse(document))
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.token_count <= 16 for chunk in chunks))

    def test_embedding_requests_are_batched(self) -> None:
        embedding = LimitedBatchEmbedding()
        service = RagIngestionService(
            InMemoryHybridIndex("batch-rag"),
            embedding,
            chunker=HeadingAwareChunker(target_tokens=16, overlap_tokens=2),
            embedding_batch_size=2,
        )
        service.ingest(
            source(
                document_id="batch-document",
                content="# Rules\n"
                + "\n\n".join(
                    f"rule {index} alpha beta gamma delta epsilon zeta eta theta"
                    for index in range(8)
                ),
            )
        )
        self.assertGreater(len(embedding.batch_sizes), 1)
        self.assertTrue(all(size <= 2 for size in embedding.batch_sizes))

    def test_oversized_atomic_block_is_rejected_before_embedding(self) -> None:
        service = RagIngestionService(
            InMemoryHybridIndex("atomic-rag"),
            TinyContextEmbedding("embedding-test", "rev-1", 32),
            chunker=HeadingAwareChunker(target_tokens=16, overlap_tokens=4),
        )
        content = "# API\n```cpp\n" + " ".join(f"token{index}" for index in range(20)) + "\n```"
        with self.assertRaises(RagServiceError) as context:
            service.prepare(source(document_id="oversized-code", content=content))
        self.assertEqual(context.exception.code, "RAG_INVALID_REQUEST")

    def test_checksum_idempotency_and_explicit_update_precondition(self) -> None:
        original = source()
        first = self.service.ingest(original)
        duplicate = self.service.ingest(original)
        self.assertEqual(first.action, "indexed")
        self.assertEqual(duplicate.action, "unchanged")
        self.assertEqual(first.generation, duplicate.generation)

        changed = replace(original, content="# Safety\nThe approved clearance is now 7 mm.")
        with self.assertRaises(RagServiceError) as conflict:
            self.service.ingest(changed)
        self.assertEqual(conflict.exception.code, "RAG_INVALID_REQUEST")

        updated = self.service.update(changed, expected_checksum=original.checksum_sha256)
        self.assertEqual(updated.action, "updated")
        self.assertGreater(updated.generation, first.generation)
        self.assertEqual([item.generation for item in self.backend.generation_history()], [0, 1, 2])

    def test_delete_is_idempotent_and_version_scoped(self) -> None:
        self.service.ingest(source(version="v2", document_id="planning-rules-v2"))
        self.service.ingest(source(version="v3", document_id="planning-rules-v3"))
        deleted = self.service.delete("planning-rules-v2", "v2")
        duplicate = self.service.delete("planning-rules-v2", "v2")
        self.assertEqual(deleted.action, "deleted")
        self.assertEqual(duplicate.action, "unchanged")
        self.assertEqual(self.backend.health().document_count, 1)

    def test_missing_security_identity_and_reserved_metadata_are_rejected(self) -> None:
        invalid = (
            {"access_scopes": ()},
            {"owner": ""},
            {"version": ""},
            {"metadata": {"status": "deprecated"}},
            {"document_id": "bad identity with spaces"},
            {"title": "forged\nsecond line"},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                source(**override)

    def test_metadata_depth_and_size_limits_are_enforced(self) -> None:
        nested = value = {}
        for _ in range(10):
            child = {}
            value["child"] = child
            value = child
        with self.assertRaisesRegex(ValueError, "nesting"):
            source(metadata=nested)
        with self.assertRaisesRegex(ValueError, "4096"):
            source(metadata={"category": "x" * 5000})

    def test_embedding_dimension_change_blocks_activation(self) -> None:
        self.service.ingest(source())
        generation = self.backend.descriptor().generation
        incompatible = RagIngestionService(
            self.backend,
            DeterministicEmbeddingBackend("embedding-test", "rev-1", 64),
        )
        with self.assertRaises(RagServiceError) as context:
            incompatible.ingest(source(document_id="other-doc", version="v1"))
        self.assertEqual(context.exception.code, "RAG_INDEX_INCOMPATIBLE")
        self.assertEqual(self.backend.descriptor().generation, generation)

    def test_embedding_instruction_change_requires_a_new_index_generation(self) -> None:
        self.service.ingest(source())
        incompatible = RagIngestionService(
            self.backend,
            DifferentInstructionEmbedding("embedding-test", "rev-1", 32),
        )
        with self.assertRaises(RagServiceError) as context:
            incompatible.ingest(source(document_id="other-manifest", version="v1"))
        self.assertEqual(context.exception.code, "RAG_INDEX_INCOMPATIBLE")

    def test_partial_rebuild_failure_keeps_old_live_generation(self) -> None:
        self.service.ingest(source())
        generation = self.backend.descriptor().generation
        failing_service = RagIngestionService(
            self.backend,
            FailingEmbedding("embedding-test", "rev-1", 32),
        )
        with self.assertRaises(RagServiceError) as context:
            failing_service.rebuild(
                (
                    source(document_id="new-good", version="v1"),
                    source(document_id="new-bad", version="v1", content="# Failure\nFAIL_EMBED"),
                )
            )
        self.assertEqual(context.exception.code, "RAG_EMBEDDING_UNAVAILABLE")
        self.assertEqual(self.backend.descriptor().generation, generation)
        self.assertEqual(self.backend.health().document_count, 1)

    def test_index_manifest_records_embedding_instructions_and_limits(self) -> None:
        self.service.ingest(source())
        descriptor = self.backend.descriptor()
        self.assertEqual(descriptor.embedding_model, "embedding-test")
        self.assertEqual(descriptor.embedding_revision, "rev-1")
        self.assertEqual(descriptor.embedding_dimension, 32)
        self.assertTrue(descriptor.query_instruction)
        self.assertTrue(descriptor.document_instruction)
        self.assertTrue(descriptor.vectors_normalized)
        self.assertEqual(descriptor.tokenizer_revision, "rag-regex-tokenizer-v1")
        self.assertEqual(descriptor.max_input_tokens, 8192)


if __name__ == "__main__":
    unittest.main()
