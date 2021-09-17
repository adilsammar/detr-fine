# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
import albumentations as A

import random
import numpy as np
from PIL import Image


class ConstructionDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset_type):
        super(ConstructionDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertConstructionPolysToMask(return_masks)
        self.dataset_type = dataset_type

    def get_random_set(self, idx):
        collage_images = list(random.sample(range(len(self.ids)), 3))

        while idx in collage_images:
            collage_images = list(random.sample(range(len(self.ids)), 3))

        collage_images.append(idx)

        targets = {i: [] for i in collage_images}
        images = {i: [] for i in collage_images}

        return collage_images, targets, images

    def get_images(self, idx):
        collage_images, targets, images = self.get_random_set(idx)

        for imid in collage_images:
            image, target = super(ConstructionDetection, self).__getitem__(imid)
            target = {'image_id': imid, 'annotations': target}
            image, target = self.prepare(image, target)
            targets[imid] = target
            images[imid] = image

        return images, targets

    def __getitem__(self, idx):
        _flip = flip_coin()
        image_id = self.ids[idx]

        if _flip or self.dataset_type == 'val':
            img, target = super(ConstructionDetection, self).__getitem__(idx)
        else:
            images, targets = self.get_images(idx)
            img, target = prepare_collage(images, targets)

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            if self.dataset_type == 'val':
                img, target = self._transforms(img, target)
            elif _flip:
                img, target = self._transforms[0](img, target)
            else:
                img, target = self._transforms[1](img, target)
        return img, target


def flip_coin():
    if torch.rand(1) > 1.0:
        return True
    else:
        return False


def prepare_collage(imgs, targets):
    idxs = imgs.keys()

    bbs = {i: [] for i in idxs}
    cats = {i: [] for i in idxs}

    collage_target = []

    for i in idxs:
        targets[i]["boxes"][:, 2:] -= targets[i]["boxes"][:, :2]
        bbs[i] = targets[i]["boxes"].int().tolist()
        cats[i] = targets[i]["labels"].int().tolist()

    trans_imgs = []
    trans_bbs = torch.tensor([])
    trans_cats = []

    transform = A.Compose(
        [A.SmallestMaxSize(max_size=400), A.RandomCrop(width=300, height=300)],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.2),
    )

    for i in idxs:
        image = np.array(imgs[i])

        transformed = transform(image=image, bboxes=bbs[i], category_ids=cats[i])

        trans_imgs.append(transformed)
        bb_tensor = torch.tensor(transformed['bboxes'])

        if len(bb_tensor) > 0:

            if i == 1:
                bb_tensor[:, 1] += 300
            if i == 2:
                bb_tensor[:, 0] += 300
            if i == 3:
                bb_tensor[:, 0] += 300
                bb_tensor[:, 1] += 300

            trans_bbs = torch.cat([trans_bbs, bb_tensor], dim=0)
            trans_cats += transformed['category_ids']

    collage_image = Image.fromarray(torch.cat([
        torch.cat([
            torch.tensor(trans_imgs[0]['image']),
            torch.tensor(trans_imgs[1]['image'])
        ], dim=0),
        torch.cat([
            torch.tensor(trans_imgs[2]['image']),
            torch.tensor(trans_imgs[3]['image'])
        ], dim=0)
    ], dim=1).detach().numpy())

    for bb, cid, ar in zip(trans_bbs, trans_cats, trans_bbs[:, 2] * trans_bbs[:, 3]):
        collage_target.append({
            'bbox': bb.tolist(),
            'category_id': cid,
            'area': ar
        })

    return collage_image, collage_target


def convert_construction_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertConstructionPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_construction_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_construction_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Construction path {root} does not exist'
    mode = 'coco'
    PATHS = {
        "train": (root / "images", root / f'{mode}.json'),
        "val": (root / "images", root / f'val_{mode}.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    if image_set == 'train':
        dataset = ConstructionDetection(img_folder, ann_file, transforms=(make_construction_transforms(
            'train'), make_construction_transforms('val')), return_masks=args.masks, dataset_type=image_set)
    else:
        dataset = ConstructionDetection(img_folder, ann_file, transforms=make_construction_transforms(
            image_set), return_masks=args.masks, dataset_type=image_set)
    return dataset
