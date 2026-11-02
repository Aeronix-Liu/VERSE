"""SPAR-Bench rubrics for DGE ground-truth computation."""

from typing import List

from .base_rubric import BaseRubric


class SPARDepthOcRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_depth_prediction_oc",
            task_id="spar_depth_prediction_oc",
            description="SPAR depth prediction (one-camera anchor)",
            rubric_alias_ids=("rubric_spar_depth_prediction_oc",),
            task_alias_ids=("spar_depth_prediction_oc", "depth_prediction_oc"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract the two referenced objects from the question",
                      required_params=["question"])
        self.add_step("spar_depth_tool", "Compute camera-frame depth of both objects",
                      required_params=["scene_id", "frame_id", "frame_metadata",
                                       "object_labels", "mode"])


class SPARDepthOoRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_depth_prediction_oo",
            task_id="spar_depth_prediction_oo",
            description="SPAR depth difference between two objects",
            rubric_alias_ids=("rubric_spar_depth_prediction_oo",),
            task_alias_ids=("spar_depth_prediction_oo", "depth_prediction_oo"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract the two referenced objects",
                      required_params=["question"])
        self.add_step("spar_depth_tool", "Compute depth of both objects",
                      required_params=["scene_id", "frame_id", "frame_metadata",
                                       "object_labels", "mode"])


class SPARDistanceOcRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_distance_prediction_oc",
            task_id="spar_distance_prediction_oc",
            description="SPAR camera-to-object distance",
            rubric_alias_ids=("rubric_spar_distance_prediction_oc",),
            task_alias_ids=("spar_distance_prediction_oc", "distance_prediction_oc"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract the referenced object",
                      required_params=["question"])
        self.add_step("spar_distance_tool", "Compute camera-to-object Euclidean distance",
                      required_params=["scene_id", "frame_id", "frame_metadata",
                                       "object_labels", "mode"])


class SPARDistanceOoRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_distance_prediction_oo",
            task_id="spar_distance_prediction_oo",
            description="SPAR object-to-object distance",
            rubric_alias_ids=("rubric_spar_distance_prediction_oo",),
            task_alias_ids=("spar_distance_prediction_oo", "distance_prediction_oo"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract the two referenced objects",
                      required_params=["question"])
        self.add_step("spar_distance_tool", "Compute object-to-object Euclidean distance",
                      required_params=["scene_id", "frame_id", "frame_metadata",
                                       "object_labels", "mode"])


class SPARDistanceInferCenterRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_distance_infer_center_oo",
            task_id="spar_distance_infer_center_oo",
            description="SPAR which object is farther from anchor",
            rubric_alias_ids=("rubric_spar_distance_infer_center_oo",),
            task_alias_ids=("spar_distance_infer_center_oo", "distance_infer_center_oo"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract anchor and two option objects",
                      required_params=["question"])
        self.add_step("spar_distance_infer_tool", "Determine which option is farther",
                      required_params=["frame_metadata", "anchor_label",
                                       "option_a_label", "option_b_label"])


class SPARSpatialRelationRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_obj_spatial_relation",
            task_id="spar_obj_spatial_relation",
            description="SPAR spatial relation between two objects",
            rubric_alias_ids=("rubric_spar_obj_spatial_relation",),
            task_alias_ids=("spar_obj_spatial_relation",
                            "obj_spatial_relation_oo",
                            "obj_spatial_relation_oc_mv",
                            "obj_spatial_relation_oo_mv"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract reference and target objects",
                      required_params=["question"])
        self.add_step("spar_spatial_relation_tool", "Classify spatial relation",
                      required_params=["scene_id", "frame_id", "frame_metadata",
                                       "ref_label", "target_label", "mode"])


class SPARPositionMatchingRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_position_matching",
            task_id="spar_position_matching",
            description="SPAR position matching across frames",
            rubric_alias_ids=("rubric_spar_position_matching",),
            task_alias_ids=("spar_position_matching", "position_matching"),
        )

    def _build_steps(self):
        self.add_step("vlm_tool", "Extract the target object from the question",
                      required_params=["question"])
        self.add_step("spar_position_matching_tool", "Project bbox into frame 2 and match",
                      required_params=["scene_id", "frame_id_1", "frame_id_2",
                                       "frame_metadata_1", "frame_metadata_2",
                                       "target_label", "candidate_bboxes"])


class SPARViewChangeInferRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_view_change_infer",
            task_id="spar_view_change_infer",
            description="SPAR camera movement between two frames",
            rubric_alias_ids=("rubric_spar_view_change_infer",),
            task_alias_ids=("spar_view_change_infer", "view_change_infer"),
        )

    def _build_steps(self):
        self.add_step("spar_view_change_infer_tool", "Compute camera delta",
                      required_params=["scene_id", "frame_id_1", "frame_id_2"])


class SPARCameraMotionInferRubric(BaseRubric):
    def __init__(self):
        super().__init__(
            rubric_id="rubric_spar_camera_motion_infer",
            task_id="spar_camera_motion_infer",
            description="SPAR camera motion inference",
            rubric_alias_ids=("rubric_spar_camera_motion_infer",),
            task_alias_ids=("spar_camera_motion_infer", "camera_motion_infer"),
        )

    def _build_steps(self):
        self.add_step("spar_camera_motion_infer_tool", "Project camera 2 into camera 1",
                      required_params=["scene_id", "frame_id_1", "frame_id_2", "options"])


SPAR_RUBRIC_REGISTRY = {
    "spar_depth_prediction_oc": SPARDepthOcRubric,
    "spar_depth_prediction_oo": SPARDepthOoRubric,
    "spar_depth_prediction_oc_mv": SPARDepthOcRubric,
    "spar_depth_prediction_oo_mv": SPARDepthOoRubric,
    "spar_distance_prediction_oc": SPARDistanceOcRubric,
    "spar_distance_prediction_oo": SPARDistanceOoRubric,
    "spar_distance_prediction_oc_mv": SPARDistanceOcRubric,
    "spar_distance_prediction_oo_mv": SPARDistanceOoRubric,
    "spar_distance_infer_center_oo": SPARDistanceInferCenterRubric,
    "spar_distance_infer_center_oo_mv": SPARDistanceInferCenterRubric,
    "spar_obj_spatial_relation_oo": SPARSpatialRelationRubric,
    "spar_obj_spatial_relation_oc_mv": SPARSpatialRelationRubric,
    "spar_obj_spatial_relation_oo_mv": SPARSpatialRelationRubric,
    "spar_position_matching": SPARPositionMatchingRubric,
    "spar_view_change_infer": SPARViewChangeInferRubric,
    "spar_camera_motion_infer": SPARCameraMotionInferRubric,
}
