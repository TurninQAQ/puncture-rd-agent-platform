"""The ten stable tool definitions exposed to MCP/Agent adapters."""

from contracts.tool_inputs import TOOL_REQUEST_TYPES
from contracts.tool_outputs import TOOL_RESULT_TYPES

from .registry import ToolDefinition


_DESCRIPTIONS = {
    "inspect_case_metadata": "Inspect CT/artifact availability, checksum state, and geometry compatibility.",
    "convert_mcs_to_nifti": "Convert a Mimics MCS label export into an nnU-Net-compatible NIfTI labelmap.",
    "validate_label_schema": "Validate observed label values against the versioned project label schema.",
    "run_segmentation": "Run the selected TensorRT segmentation model on a CT artifact.",
    "validate_segmentation_result": "Check segmentation geometry, required labels, components, and volume thresholds.",
    "extract_skin_surface": "Extract a thin outer skin mask used for entry-point and penetration checks.",
    "generate_candidate_paths": "Generate geometrically valid skin-to-target needle candidates.",
    "evaluate_path_safety": "Evaluate full-path safety envelopes against dangerous-organ masks.",
    "evaluate_intraoperative_risk": "Evaluate current needle-tip warning/stop and organ-entry risks.",
    "verify_skin_penetration": "Use 3D ray sampling to determine penetration or suspected skin slip.",
}

_TIMEOUTS_MS = {
    "inspect_case_metadata": 10_000,
    "convert_mcs_to_nifti": 60_000,
    "validate_label_schema": 15_000,
    "run_segmentation": 120_000,
    "validate_segmentation_result": 30_000,
    "extract_skin_surface": 30_000,
    "generate_candidate_paths": 60_000,
    "evaluate_path_safety": 60_000,
    "evaluate_intraoperative_risk": 10_000,
    "verify_skin_penetration": 10_000,
}

_WRITE_TOOLS = {
    "convert_mcs_to_nifti",
    "run_segmentation",
    "extract_skin_surface",
    "generate_candidate_paths",
}

TOOL_DEFINITIONS = {
    name: ToolDefinition(
        name=name,
        version="1.0.0",
        request_type=TOOL_REQUEST_TYPES[name],
        result_type=TOOL_RESULT_TYPES[name],
        description=_DESCRIPTIONS[name],
        read_only=name not in _WRITE_TOOLS,
        destructive=False,
        idempotent=True,
        open_world=False,
        default_timeout_ms=_TIMEOUTS_MS[name],
    )
    for name in TOOL_REQUEST_TYPES
}
