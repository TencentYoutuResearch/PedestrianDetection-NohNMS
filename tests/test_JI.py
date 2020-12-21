import os
import json

from detectron2.evaluation.crowdhuman_evaluation import JI_evaluate

fpath = "/data1/penghaozhou/workspace/PedestrianDetection/detectron2/datasets/crowdhuman/annotations/annotation_val.odgt"
with open(fpath, "r") as fid:
    lines = fid.readlines()
gts = [json.loads(line.strip("\n")) for line in lines]

result_file = "/data1/penghaozhou/workspace/Experiments/detectron2/Crowdhuman/Overlaperceptive/faster_rcnn_R_50_FPN_baseline_iou0.5_pure_overlap_perceptive_nms_uniform_reg_divisor_threshold_0.4/inference/submission.txt"
# "/data1/penghaozhou/workspace/Experiments/detectron2/CrowdhumanCompetition/faster_rcnn_R_50_FPN_baseline_iou0.5_mst_600_800_freeze_at_1/inference/submission_0.7.txt"
with open(result_file, "r") as fid:
    lines = fid.readlines()
submit_results = [json.loads(line.strip("\n")) for line in lines]
result = JI_evaluate(submit_results, gts)
print(result)
