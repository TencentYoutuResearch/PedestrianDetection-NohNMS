import torch


EXTRA = ['proposal_generator.anchor_generator.cell_anchors.0',
 'proposal_generator.anchor_generator.cell_anchors.1',
 'proposal_generator.anchor_generator.cell_anchors.2',
 'proposal_generator.anchor_generator.cell_anchors.3',
 'proposal_generator.anchor_generator.cell_anchors.4',]

crowddet_models = torch.load("/data1/penghaozhou/workspace/PretrainModels/dump-1.pth")['state_dict']
detectron_models = torch.load("/data1/penghaozhou/workspace/Experiments/detectron2/Crowdhuman/faster_rcnn_R_50_FPN_baseline_iou0.5_align_crowddet_without_bn/model_0000799.pth")['model']

crowddet_keys = list(crowddet_models.keys())
detectron_keys = list(detectron_models.keys())
crowddet_bottom_up = list()
crowddet_fpn = list()
for key in crowddet_keys:
    if "FPN.bottom_up" in key:
        crowddet_bottom_up.append(key)
    elif 'FPN.fpn' in key:
        crowddet_fpn.append(key)

detectron_bottom_up = list()
detectron_fpn = list()

for key in detectron_keys:
    if "backbone.bottom_up" in key:
        detectron_bottom_up.append(key)
    elif "backbone.fpn" in key:
        detectron_fpn.append(key)

RCNN =  [[
 'RCNN.fc1.weight',
 'RCNN.fc1.bias',
 'RCNN.fc2.weight',
 'RCNN.fc2.bias',
 'RCNN.pred_cls.weight',
 'RCNN.pred_cls.bias',
 'RCNN.pred_delta.weight',
 'RCNN.pred_delta.bias'],
 ['roi_heads.box_head.fc1.weight',
 'roi_heads.box_head.fc1.bias',
 'roi_heads.box_head.fc2.weight',
 'roi_heads.box_head.fc2.bias',
 'roi_heads.box_predictor.cls_score.weight',
 'roi_heads.box_predictor.cls_score.bias',
 'roi_heads.box_predictor.bbox_pred.weight',
 'roi_heads.box_predictor.bbox_pred.bias']]

RPN = [['RPN.rpn_conv.weight',
 'RPN.rpn_conv.bias',
 'RPN.rpn_cls_score.weight',
 'RPN.rpn_cls_score.bias',
 'RPN.rpn_bbox_offsets.weight',
 'RPN.rpn_bbox_offsets.bias'],
 [
 'proposal_generator.rpn_head.conv.weight',
 'proposal_generator.rpn_head.conv.bias',
 'proposal_generator.rpn_head.objectness_logits.weight',
 'proposal_generator.rpn_head.objectness_logits.bias',
 'proposal_generator.rpn_head.anchor_deltas.weight',
 'proposal_generator.rpn_head.anchor_deltas.bias']]

convert_model = dict()
for i, key in enumerate(crowddet_bottom_up):
    convert_model[detectron_bottom_up[i]] = crowddet_models[key]
for i, key in enumerate(crowddet_fpn):
    convert_model[detectron_fpn[i]] = crowddet_models[key]
    
for i in range(len(RCNN[0])):
    convert_model[RCNN[1][i]] = crowddet_models[RCNN[0][i]]

for i in range(len(RPN[0])):
    convert_model[RPN[1][i]] = crowddet_models[RPN[0][i]]

torch.save({'model':convert_model}, "/data1/penghaozhou/workspace/PretrainModels/convert_crowddet_pretrain_model.pth")