# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.data

from PIL import Image
from panopticapi.utils import rgb2id

from utils.utils import masks_to_boxes
from dataset.utils import make_coco_transforms

city2int = {
    "aachen": 0,
    "bremen": 1,
    "darmstadt": 2,
    "erfurt": 3,
    "hanover": 4,
    "krefeld": 5,
    "strasbourg": 6,
    "tubingen": 7,
    "weimar": 8,
    "bochum": 9,
    "cologne": 10,
    "dusseldorf": 11,
    "hamburg": 12,
    "jena": 13,
    "monchengladbach": 14,
    "stuttgart": 15,
    "ulm": 16,
    "zurich": 17,
    "frankfurt": 18,
    "lindau": 19,
    "munster":20,
    "berlin": 21,
    "bielefeld": 22,
    "bonn": 23,
    "leverkusen": 24,
    "mainz": 25,
    "munich": 26
}

int2city = {v: k for k, v in city2int.items()}

def imgid2int(id):
    city, f, s = id.split('_')
    return int(int(s) + int(f)*1e6 + city2int[city]*1e12)

def int2imgid(num):
    cityn = num // int(1e12)
    f = (num - int(cityn * 1e12)) // int(1e6)
    s = num % int(1e6)
    return int2city[cityn]+'_'+str(f).zfill(6)+'_'+str(s).zfill(6)

class CityscapesPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        city = ann_info['image_id'].split('_')[0]
        img_path = Path(self.img_folder) / city / (ann_info['image_id'] + "_leftImg8bit.png")
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([imgid2int(ann_info['image_id'] if "image_id" in ann_info else ann_info["id"])])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco['images'])

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width

def build_cityscapes_panoptic(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided path {ann_folder_root} does not exist'
    
    ann_file = {
        "train": "/content/drive/MyDrive/cityscapes/gtFine/cityscapes_panoptic_train.json",
        "val": "/content/drive/MyDrive/cityscapes/gtFine/cityscapes_panoptic_val.json"
    }

    img_folder_path = {
        "train": "/content/drive/MyDrive/cityscapes/leftImg8bit/train",
        "val": "/content/drive/MyDrive/cityscapes/leftImg8bit/val"
    }

    ann_folder = {
        "train": "/content/drive/MyDrive/cityscapes/gtFine/cityscapes_panoptic_train",
        "val": "/content/drive/MyDrive/cityscapes/gtFine/cityscapes_panoptic_val"
    }

    dataset = CityscapesPanoptic(img_folder_path[image_set], ann_folder[image_set], ann_file[image_set],
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset

def build_dataset(image_set, args):
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        return build_cityscapes_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
