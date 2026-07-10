"""Real implementation entry points.

Replace one function at a time with an adapter to company C++/TensorRT code.
Do not change signatures; the MCP layer and contract tests depend on them.
"""

from contracts.common import ToolResponseEnvelope
from contracts.tool_inputs import (
    ConvertMcsToNiftiRequest,
    EvaluateIntraoperativeRiskRequest,
    EvaluatePathSafetyRequest,
    ExtractSkinSurfaceRequest,
    GenerateCandidatePathsRequest,
    InspectCaseMetadataRequest,
    RunSegmentationRequest,
    ValidateLabelSchemaRequest,
    ValidateSegmentationResultRequest,
    VerifySkinPenetrationRequest,
)
from contracts.tool_outputs import (
    CandidatePathGenerationResult,
    CaseMetadataResult,
    IntraoperativeRiskResult,
    LabelSchemaValidationResult,
    McsToNiftiResult,
    PathSafetyEvaluationResult,
    SegmentationInferenceResult,
    SegmentationValidationResult,
    SkinPenetrationResult,
    SkinSurfaceExtractionResult,
)


def _todo(name: str) -> None:
    raise NotImplementedError(f"Implement {name} according to specs/tools/{name.replace('_', '-')}.md")


def inspect_case_metadata(request: InspectCaseMetadataRequest) -> ToolResponseEnvelope[CaseMetadataResult]:
    _todo("inspect_case_metadata")


def convert_mcs_to_nifti(request: ConvertMcsToNiftiRequest) -> ToolResponseEnvelope[McsToNiftiResult]:
    _todo("convert_mcs_to_nifti")


def validate_label_schema(request: ValidateLabelSchemaRequest) -> ToolResponseEnvelope[LabelSchemaValidationResult]:
    _todo("validate_label_schema")


def run_segmentation(request: RunSegmentationRequest) -> ToolResponseEnvelope[SegmentationInferenceResult]:
    _todo("run_segmentation")


def validate_segmentation_result(
    request: ValidateSegmentationResultRequest,
) -> ToolResponseEnvelope[SegmentationValidationResult]:
    _todo("validate_segmentation_result")


def extract_skin_surface(request: ExtractSkinSurfaceRequest) -> ToolResponseEnvelope[SkinSurfaceExtractionResult]:
    _todo("extract_skin_surface")


def generate_candidate_paths(
    request: GenerateCandidatePathsRequest,
) -> ToolResponseEnvelope[CandidatePathGenerationResult]:
    _todo("generate_candidate_paths")


def evaluate_path_safety(request: EvaluatePathSafetyRequest) -> ToolResponseEnvelope[PathSafetyEvaluationResult]:
    _todo("evaluate_path_safety")


def evaluate_intraoperative_risk(
    request: EvaluateIntraoperativeRiskRequest,
) -> ToolResponseEnvelope[IntraoperativeRiskResult]:
    _todo("evaluate_intraoperative_risk")


def verify_skin_penetration(request: VerifySkinPenetrationRequest) -> ToolResponseEnvelope[SkinPenetrationResult]:
    _todo("verify_skin_penetration")
