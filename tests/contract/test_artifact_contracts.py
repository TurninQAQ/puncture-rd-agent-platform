from __future__ import annotations

import unittest
from dataclasses import fields

from contracts.artifacts import ArtifactPublicView
from contracts.enums import ArtifactStatus


class ArtifactContractTests(unittest.TestCase):
    def test_canonical_status_values_are_frozen(self) -> None:
        self.assertEqual(
            ["AVAILABLE", "PENDING", "INVALID", "MISSING"],
            [item.value for item in ArtifactStatus],
        )

    def test_public_view_cannot_expose_storage_location(self) -> None:
        names = {item.name for item in fields(ArtifactPublicView)}
        forbidden = {"uri", "internal_uri", "checksum_sha256", "metadata", "parent_artifact_ids"}
        self.assertFalse(names.intersection(forbidden))


if __name__ == "__main__":
    unittest.main()
