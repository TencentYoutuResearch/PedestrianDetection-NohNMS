# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved
"""
@contact:
penghaozhou@tencent.com
"""
import logging
import os
import numpy as np

import cv2
import datetime
from scipy import io
import json

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse crowdhuman-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["register_crowdhuman_instances", "load_crowdhuman"]


def load_crowdhuman(anno_file, image_dir):
    """
    Return dataset dict.
    """
    anno_lines = open(anno_file, "r").readlines()
    annos = [json.loads(line.strip()) for line in anno_lines]

    image_num = len(annos)
    logger.info("Loaded {} images in CrowdHuman from {}".format(image_num, anno_file))

    dataset_dicts = []
    ignore_instances = 0
    instances = 0
    for img_id, anno in enumerate(annos):
        record = {}
        record["image_id"] = img_id + 1
        record["ID"] = anno["ID"]
        record["file_name"] = os.path.join(image_dir, anno["ID"] + ".jpg")

        objs = []
        for gt_box in anno["gtboxes"]:
            if gt_box["fbox"][2] < 0 or gt_box["fbox"][3] < 0:
                continue
            obj = {}
            obj["bbox"] = gt_box["fbox"]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) != 0:
                obj["category_id"] = -1
                ignore_instances += 1
            else:
                obj["category_id"] = 0
            instances += 1

            vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(
                (gt_box["fbox"][2] * gt_box["fbox"][3])
            )
            obj["vis_ratio"] = vis_ratio

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    logger.info(
        "Loaded {} instances and {} ignore instances in CrowdHuman from {}".format(
            instances, ignore_instances, anno_file
        )
    )

    return dataset_dicts


def register_crowdhuman_instances(name, metadata, anno_file, image_dir, val_json_files):
    """
    Register CityPersons dataset.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_crowdhuman(anno_file, image_dir))

    if not isinstance(val_json_files, list):
        val_json_files = [val_json_files]

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=val_json_files,
        anno_file=anno_file,
        image_dir=image_dir,
        evaluator_type="crowdhuman",
        **metadata,
    )
    if "val" in name:
        for val_json_file in val_json_files:
            if not os.path.exists(val_json_file):
                is_clip = "clip" in val_json_file
                convert_to_coco_dict(anno_file, image_dir, val_json_file, is_clip=is_clip)


def convert_to_coco_dict(anno_file, image_dir, json_file, is_clip=True):
    from tqdm import tqdm

    anno_lines = open(anno_file, "r").readlines()
    annos = [json.loads(line.strip()) for line in anno_lines]

    print("Converting dataset dicts into COCO format")

    images = []
    annotations = []
    outside_num, clip_num = 0, 0
    for img_id, anno in tqdm(enumerate(annos)):
        filename = os.path.join(image_dir, anno["ID"] + ".jpg")
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        image = {"id": img_id + 1, "file_name": filename, "height": height, "width": width}
        images.append(image)

        for gt_box in anno["gtboxes"]:
            annotation = {}
            x1, y1, w, h = gt_box["fbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            annotation["id"] = len(annotations) + 1
            annotation["image_id"] = image["id"]

            annotation["area"] = gt_box["fbox"][2] * gt_box["fbox"][3]
            if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) == 1:
                annotation["ignore"] = 1
            elif outside(bbox, height, width):
                annotation["ignore"] = 1
                outside_num += 1
            elif is_clip and (
                (bbox[0] < 0) or (bbox[1] < 0) or (bbox[2] > width) or (bbox[3] > height)
            ):
                bbox = clip_bbox(bbox, [height, width])
                clip_num += 1
                annotation["ignore"] = 0
            else:
                annotation["ignore"] = 0

            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]

            annotation["category_id"] = 1
            annotation["bbox"] = [round(float(x), 3) for x in bbox]
            annotation["height"] = annotation["bbox"][3]
            vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(annotation["area"])
            annotation["vis_ratio"] = vis_ratio
            annotation["iscrowd"] = 0
            annotations.append(annotation)

    print("outside num: {}, clip num: {}".format(outside_num, clip_num))
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated CrowdHuman json file for Detectron2.",
    }

    categories = [{"id": 1, "name": "pedestrian"}]

    coco_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    try:
        json.dump(coco_dict, open(json_file, "w"))
    except:
        print("json dump falied in crowdhuman convert processing.")
        from IPython import embed

        embed()


def clip_bbox(bbox, box_size):
    height, width = box_size
    bbox[0] = np.clip(bbox[0], 0, width)
    bbox[1] = np.clip(bbox[1], 0, height)
    bbox[2] = np.clip(bbox[2], 0, width)
    bbox[3] = np.clip(bbox[3], 0, height)
    return bbox


def outside(bbox, height, width):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if cx < 0 or cx > width or cy < 0 or cy > height:
        return True
    else:
        return False
