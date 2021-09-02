# Import Required Libs

import os
import sys
from categories_meta import COCO_CATEGORIES, COCO_NAMES
from panopticapi.utils import id2rgb, rgb2id
import panopticapi
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import math
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
import itertools
import seaborn as sns
palette = itertools.cycle(sns.color_palette())

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy as np
torch.set_grad_enabled(False)

import convert_to_coco


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Workaround for torch issue to load detr lib ------
os.system('git clone https://github.com/facebookresearch/detr.git')
sys.path.append(os.path.join(os.getcwd(), "detr/"))

# ------------------------------------------------------


# Load detr model
model, postprocessor = torch.hub.load('detr', 'detr_resnet101_panoptic',
                                      source='local', pretrained=True, return_postprocessor=True, num_classes=250)
# Convert to eval mode
model.eval()


# Load Test Image
# url = "http://images.cocodataset.org/val2017/000000281759.jpg"
url = "https://img2.goodfon.com/wallpaper/nbig/6/f4/rossiya-priroda-pole-zelen.jpg"
# stream = requests.get(url, stream=True).raw
# imo = Image.open(stream)
imo = Image.open('./img_099.png')
im = imo.resize((800, 600))

h, w, c = np.array(imo).shape

if c == 4:
    imo = Image.open('./img_099.png').convert('RGB')
    im = imo.resize((800, 600))
    h, w, c = np.array(imo).shape

# print("Height", h, "Width", w)

# Apply transform and convert image to batch
# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)  # [h, w, c] -> [1, c, h, w]

# print(img.shape)

# Generate output for image
out = model(img)

# out.keys()
## dict_keys(['pred_logits', 'pred_boxes', 'pred_masks'])

# print(out['pred_logits'].shape, out['pred_boxes'].shape)
## (torch.Size([batch, keys, classes]), torch.Size([batch, keys, bb(4)]))

# Generate score
# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Keep only ones above threshold
pred_logits, pred_boxes = out["pred_logits"][keep][:, :len(
    COCO_NAMES)-1], out["pred_boxes"][keep]

# print(pred_logits.shape, pred_boxes.shape)
# (torch.Size([above_threshold_preductions, classes(200)]), torch.Size([above_threshold_preductions, bb(4)]))


## Draw predicted BB
# im2 = imo.copy()
# drw = ImageDraw.Draw(im2)
# for logits, box in zip(pred_logits, pred_boxes):
#     cls = logits.argmax()
#     if cls >= 200:
#         continue
#     label = COCO_NAMES[cls]
#     box = box.cpu() * torch.Tensor([w, h, w, h])
#     x, y, wbb, hbb = box
#     x0, x1 = x-wbb//2, x+wbb//2
#     y0, y1 = y-hbb//2, y+hbb//2
#     drw.rectangle([x0, y0, x1, y1], outline='red', width=3)
#     drw.text((x0, y0), label, fill='white')

# im2.show()

## Plot all the remaining masks (throshelded)
# ncols = 2
# fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(10, 7))
# for line in axs:
#     for a in line:
#         a.axis('off')
# for i, mask in enumerate(out["pred_masks"][keep]):
#     ax = axs[i // ncols, i % ncols]
#     ax.imshow(mask, cmap="cividis")
#     ax.axis('off')
# fig.tight_layout()


# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string'])).resize((w, h), Image.NEAREST)
(wp, hp) = panoptic_seg.size
# print("Height Pan", hp, "Width Pan", wp)
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb2id(panoptic_seg)

# print(result['segments_info'], panoptic_seg.shape, panoptic_seg_id.shape)
# For Example Image
# ([{'area': 59747, 'category_id': 184, 'id': 0, 'isthing': False},
#   {'area': 269818, 'category_id': 193, 'id': 1, 'isthing': False},
#   {'area': 505977, 'category_id': 187, 'id': 2, 'isthing': False},
#   {'area': 17258, 'category_id': 194, 'id': 3, 'isthing': False}],
#  (800, 1066, 3),
#  (800, 1066))

## Visualize Masked image
## Finally we color each mask individually
# panoptic_seg[:, :, :] = 0
# for id in range(panoptic_seg_id.max() + 1):
#   panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
# plt.figure(figsize=(10,10))
# plt.imshow(panoptic_seg)
# plt.axis('off')
# plt.show()

# Convert to binary segment
binary_masks = np.zeros((
    panoptic_seg_id.max() + 1,
    panoptic_seg_id.shape[0],
    panoptic_seg_id.shape[1]),
    dtype=np.uint8
)

import datetime
import json

ROOT_DIR = './'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/adilsammar/custom_coco_dataset",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "adilsammar",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

from categories_meta import COCO_CATEGORIES as CATEGORIES
import coco_creator_tools

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

image_id = 1

# Opening JSON file
with open('test.json') as f:
    # returns JSON object as 
    # a dictionary
    omask = json.load(f)

# Create Original Segmented Image
import overlay_custom_mask

omask_image_id = overlay_custom_mask.get_overlayed_mask((h, w), omask)

panoptic_seg_id[omask_image_id.astype(np.bool_)] = panoptic_seg_id.max() + 1

image_info = coco_creator_tools.create_image_info(image_id, url, imo.size)

coco_output["images"].append(image_info)

for ans in omask["annotations"]:
    coco_output["annotations"].append(ans)

for id in np.unique(panoptic_seg_id)[:-1]: # Skip the last one as it is for custom mappings
    binary_masks[id, :, :] = panoptic_seg_id == id
    annotation_info = convert_to_coco.main(binary_masks[id], None, image_id, result['segments_info'][id]["category_id"], result['segments_info'][id]["id"], False)
    if annotation_info is not None:
        coco_output["annotations"].append(annotation_info)

with open('{}/instances_custom.json'.format(ROOT_DIR), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)
