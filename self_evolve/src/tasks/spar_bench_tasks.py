"""SPAR-Bench task definitions for DGE.

Each task mirrors a SPAR-Bench original_question_type. The task class
defines how the Questioner's parameters are extracted from the question
text (for the Solver's agentic rollout). Ground truth is computed by the
corresponding tool in src/tools/spar_bench_tools.py.
"""

from typing import Any, Callable, Dict, List

from .base_task import BaseTask, TaskDifficulty

STRICT_EXTRACTION_NOTICE = (
    "Role: Expert Entity Extractor for 3D Scenes.\n"
    "Instruction: Extract objects from the question exactly as they appear.\n"
    "Rules:\n"
    "- Use lowercase and singular forms.\n"
    "- Keep original spelling; do not paraphrase or replace with synonyms.\n"
    "- If missing or uncertain, output null.\n"
    "- Output only the extraction result in required format.\n"
)


class SPARDepthOcTask(BaseTask):
    """depth_prediction_oc: given depth of obj_a, predict depth of obj_b."""

    def __init__(self):
        super().__init__(
            task_id="spar_depth_prediction_oc",
            task_name="SPAR Depth Prediction OC",
            description="Predict object depth given another object's depth",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_depth_prediction_oc", "depth_prediction_oc"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_depth_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata", "object_labels"]

    def get_expected_output_format(self):
        return {"answer": "float (meters)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the two objects whose depth is discussed in the question "
            "(the object whose depth is given, then the object whose depth is asked).\n"
            "Format: given_object, target_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"object_labels": [map_label(parts[0]), map_label(parts[1])]}


class SPARDepthOoTask(BaseTask):
    """depth_prediction_oo: depth difference between two objects."""

    def __init__(self):
        super().__init__(
            task_id="spar_depth_prediction_oo",
            task_name="SPAR Depth Prediction OO",
            description="Predict depth difference between two objects",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_depth_prediction_oo", "depth_prediction_oo"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_depth_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata", "object_labels"]

    def get_expected_output_format(self):
        return {"answer": "float (meters)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the two objects whose depth difference is asked.\n"
            "Format: object_a, object_b\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"object_labels": [map_label(p) for p in parts[:2]]}


class SPARDistanceOcTask(BaseTask):
    """distance_prediction_oc: camera-to-object distance."""

    def __init__(self):
        super().__init__(
            task_id="spar_distance_prediction_oc",
            task_name="SPAR Distance Prediction OC",
            description="Predict camera-to-object distance",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_distance_prediction_oc", "distance_prediction_oc"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_distance_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata", "object_labels"]

    def get_expected_output_format(self):
        return {"answer": "float (meters)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the single object whose camera distance is asked.\n"
            "Format: object_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        label = map_label(raw_extraction.strip().lower())
        return {"object_labels": [label]}


class SPARDistanceOoTask(BaseTask):
    """distance_prediction_oo: object-to-object Euclidean distance."""

    def __init__(self):
        super().__init__(
            task_id="spar_distance_prediction_oo",
            task_name="SPAR Distance Prediction OO",
            description="Predict object-to-object distance",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_distance_prediction_oo", "distance_prediction_oo"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_distance_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata", "object_labels"]

    def get_expected_output_format(self):
        return {"answer": "float (meters)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the two objects whose distance is asked.\n"
            "Format: object_a, object_b\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"object_labels": [map_label(p) for p in parts[:2]]}


class SPARDistanceInferCenterTask(BaseTask):
    """distance_infer_center_oo: which of two objects is farther from anchor."""

    def __init__(self):
        super().__init__(
            task_id="spar_distance_infer_center_oo",
            task_name="SPAR Distance Infer Center OO",
            description="Determine which object is farther from an anchor",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_distance_infer_center_oo", "distance_infer_center_oo"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_distance_infer_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata",
                "anchor_label", "option_a_label", "option_b_label"]

    def get_expected_output_format(self):
        return {"answer": "str (A or B)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract three objects from the question: the anchor object, "
            "then option A's object, then option B's object.\n"
            "Format: anchor, option_a, option_b\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 3:
            return {}
        return {"anchor_label": map_label(parts[0]),
                "option_a_label": map_label(parts[1]),
                "option_b_label": map_label(parts[2])}


class SPARSpatialRelationTask(BaseTask):
    """obj_spatial_relation_*: spatial relation between two objects."""

    def __init__(self):
        super().__init__(
            task_id="spar_obj_spatial_relation",
            task_name="SPAR Object Spatial Relation",
            description="Classify spatial relation between two objects",
            difficulty=TaskDifficulty.MEDIUM,
            task_alias_ids=("spar_obj_spatial_relation",
                            "obj_spatial_relation_oo",
                            "obj_spatial_relation_oc_mv",
                            "obj_spatial_relation_oo_mv"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_spatial_relation_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata", "ref_label", "target_label"]

    def get_expected_output_format(self):
        return {"answer": "str (A/B/C/D)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract two objects from the question: the reference object "
            "and the target object whose spatial relation is asked.\n"
            "Format: ref_object, target_object\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        parts = [p.strip() for p in raw_extraction.strip().lower().split(",")]
        if len(parts) < 2:
            return {}
        return {"ref_label": map_label(parts[0]), "target_label": map_label(parts[1])}


class SPARPositionMatchingTask(BaseTask):
    """position_matching: locate object's bbox in frame 2."""

    def __init__(self):
        super().__init__(
            task_id="spar_position_matching",
            task_name="SPAR Position Matching",
            description="Match object bbox across two frames",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("spar_position_matching", "position_matching"),
        )

    def get_required_tools(self):
        return ["vlm_tool", "spar_position_matching_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "frame_metadata_1", "frame_metadata_2",
                "target_label", "candidate_bboxes"]

    def get_expected_output_format(self):
        return {"answer": "str (A/B/C/D)", "confidence": "float"}

    def requires_llm_extraction(self):
        return True

    def build_extraction_prompt(self, question: str) -> str:
        return (
            f"{STRICT_EXTRACTION_NOTICE}"
            "Task: Extract the target object (red bbox in the first image).\n"
            "Format: object_name\n"
            f"Question: {question}\n"
            "Extraction:"
        )

    def parse_extracted_params(self, raw_extraction: str, map_label: Callable) -> Dict[str, Any]:
        return {"target_label": map_label(raw_extraction.strip().lower())}


class SPARViewChangeInferTask(BaseTask):
    """view_change_infer: describe camera movement between two frames."""

    def __init__(self):
        super().__init__(
            task_id="spar_view_change_infer",
            task_name="SPAR View Change Infer",
            description="Describe camera movement and rotation",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("spar_view_change_infer", "view_change_infer"),
        )

    def get_required_tools(self):
        return ["spar_view_change_infer_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "scene_id", "frame_id_1", "frame_id_2"]

    def get_expected_output_format(self):
        return {"answer": "str (move_..,rotate_..)", "confidence": "float"}

    def requires_llm_extraction(self):
        return False


class SPARCameraMotionInferTask(BaseTask):
    """camera_motion_infer: project camera 2 position into camera 1 image."""

    def __init__(self):
        super().__init__(
            task_id="spar_camera_motion_infer",
            task_name="SPAR Camera Motion Infer",
            description="Estimate camera 2 position in camera 1 coordinates",
            difficulty=TaskDifficulty.HARD,
            task_alias_ids=("spar_camera_motion_infer", "camera_motion_infer"),
        )

    def get_required_tools(self):
        return ["spar_camera_motion_infer_tool"]

    def get_required_input_keys(self):
        return ["question", "images", "scene_id", "frame_id_1", "frame_id_2", "options"]

    def get_expected_output_format(self):
        return {"answer": "str (A or B)", "confidence": "float"}

    def requires_llm_extraction(self):
        return False


# Multi-view variants reuse the same task class (geometry is identical;
# the only difference is how many frames the Solver sees).
SPAR_TASK_REGISTRY = {
    "spar_depth_prediction_oc": SPARDepthOcTask,
    "spar_depth_prediction_oo": SPARDepthOoTask,
    "spar_depth_prediction_oc_mv": SPARDepthOcTask,
    "spar_depth_prediction_oo_mv": SPARDepthOoTask,
    "spar_distance_prediction_oc": SPARDistanceOcTask,
    "spar_distance_prediction_oo": SPARDistanceOoTask,
    "spar_distance_prediction_oc_mv": SPARDistanceOcTask,
    "spar_distance_prediction_oo_mv": SPARDistanceOoTask,
    "spar_distance_infer_center_oo": SPARDistanceInferCenterTask,
    "spar_distance_infer_center_oo_mv": SPARDistanceInferCenterTask,
    "spar_obj_spatial_relation_oo": SPARSpatialRelationTask,
    "spar_obj_spatial_relation_oc_mv": SPARSpatialRelationTask,
    "spar_obj_spatial_relation_oo_mv": SPARSpatialRelationTask,
    "spar_position_matching": SPARPositionMatchingTask,
    "spar_view_change_infer": SPARViewChangeInferTask,
    "spar_camera_motion_infer": SPARCameraMotionInferTask,
}
