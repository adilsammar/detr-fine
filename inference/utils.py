import math
import time
import datetime
import io
import itertools

import torch

from pathlib import Path
from copy import deepcopy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from panopticapi.utils import id2rgb, rgb2id
from detectron2.utils.visualizer import Visualizer

from detr.datasets.construction import make_construction_transforms
from detr.datasets.categories_meta import id2cat, get_builtin_metadata

palette = itertools.cycle(sns.color_palette())
meta = get_builtin_metadata("construction_panoptic_separated")


def load_image(pth, fixed_height=800):
    impath = Path(pth)
    
    imo = Image.open(impath)
    
    height_percent = (fixed_height / float(imo.size[1]))
    width_size = int((float(imo.size[0]) * float(height_percent)))

    imo = imo.resize((width_size, fixed_height))
    iw, ih = imo.size
    
    return imo, iw, ih


def apply_transform(imo, iw, ih, device):
    transform = make_construction_transforms("val")

    dummy_target = {
        "size": torch.as_tensor([int(ih), int(iw)]),
        "orig_size": torch.as_tensor([int(ih), int(iw)])
    }

    image, targets = transform(imo, dummy_target)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    return image


def run_prediction(model, image, postprocessors, device, threshold=0.85):
    outputs = model.to(device)(image)
    
    postprocessors['panoptic'].threshold = threshold
    panoptic = postprocessors['panoptic'](outputs, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]

    logits = outputs["pred_logits"].cpu()
    boxes = outputs["pred_boxes"].cpu()
    masks = outputs["pred_masks"].cpu()
    
    scores = logits.softmax(-1)[..., :-1].max(-1)[0]
    
    # threshold the confidence, filter all predictions above threshod
    keep = scores > threshold
    
    return scores[keep], logits[keep], boxes[keep], masks[keep].detach().numpy(), panoptic
    

def overlay_boxes(img, iw, ih, scores, logits, boxes, debug=False):
    imn = img.copy()
    drw = ImageDraw.Draw(imn)
    font = ImageFont.load_default() # ImageFont.truetype("arial")

    for score, logit, box in zip(scores, logits, boxes):
        cat = logit.argmax()
        if cat < 1:
            continue

        label = f'{id2cat[cat.item()]} ({score:.2f})'
        box = box * torch.Tensor([iw, ih, iw, ih])

        x, y, w, h = box
        
        # x0, x1 = x-w//2, x+w//2
        # y0, y1 = y-h//2, y+h//2
        rbbox = torch.tensor([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)]).cpu()

        rbbox[0::2].clamp_(min=0, max=torch.tensor(iw))
        rbbox[1::2].clamp_(min=0, max=torch.tensor(ih))
        
        if debug:
            print(label, rbbox)

        drw.rectangle(list(rbbox), outline='red', width=3)
        # drw.text((rbbox[0]+4, rbbox[1]+2), label, fill='white')

        # get text size
        text_size = font.getsize(label)
        # set button size + 10px margins
        label_size = (text_size[0]+6, text_size[1]+6)
        # create image with correct size and black background
        label_img = Image.new('RGBA', label_size, "green")
        # put text on button with 10px margins
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((3, 3), label, font=font, fill='white')

        # put text on source image in position (x+2, y+2)
        imn.paste(label_img, (rbbox[0]+2, rbbox[1]+2))
        
    return imn


def get_panoptic_mask(panoptic):
    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(panoptic['png_string']))

    # Convert to numpy array
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()

    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = np.asarray(next(palette)) * 255
    for sid in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == sid] = np.asarray(next(palette)) * 255
        
    return panoptic_seg


def get_panoptic_overlay(imo, panoptic):
    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(panoptic['png_string']))
    pw, ph = panoptic_seg.size

    # Convert to numpy array
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()

    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)
    
    panoptic_seg_id_tensor = torch.from_numpy(panoptic_seg_id)
    segments_info = deepcopy(panoptic["segments_info"])

    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    visualize = Visualizer(np.array(imo.copy().resize((pw, ph)))[:, :, ::-1], meta, scale=1.0)
    visualize._default_font_size = 20
    visualize = visualize.draw_panoptic_seg_predictions(panoptic_seg_id_tensor, segments_info, area_threshold=0)
    overlayed = visualize.get_image()
    
    return overlayed


def get_masks(logits, masks):
    mask_array = []
    
    for logit, mask in zip(logits, masks):
        cat = logit.argmax()
        if cat < 1:
            continue

        mask_array.append({
            'mask': mask,
            'label': f'{id2cat[cat.item()]}'
        })
    
    return mask_array   
    

def get_prediction(pth, model, threshold, device, debug=False):
    start = time.time()
    
    result = {}
    
    # Load image with path provided
    imo, iw, ih = load_image(pth)
    
    result["original_image"] = imo
    
    # Apply transform to normalize and convert to tensor
    image = apply_transform(imo, iw, ih, device)

    # Run prediction and threshold output
    scores, logits, boxes, masks, panoptic = run_prediction(model, image, postprocessors, device, threshold)
    
    result["boxed_image"] = overlay_boxes(imo, iw, ih, scores, logits, boxes, debug=debug)
    
    result["mask_images"] = get_masks(logits, masks)
    
    result["panoptic_mask"] = get_panoptic_mask(panoptic)
    
    result["panoptic_image"] = get_panoptic_overlay(imo, panoptic)
    
    print(f"Time Taken: {datetime.timedelta(seconds=int(time.time() - start))}")
    
    return result, logits, boxes, masks # keep, pred_logits, pred_masks.detach().numpy(), imn, result_panoptic


def visualize_masks(masks):
    # Plot all the remaining masks
    if len(masks) == 1:
        plt.imshow(masks[0]["mask"], cmap="cividis")
        # plt.set_title(f'{id2cat[pred_logits[1].argmax().item()]}', {'fontsize': 15})
        plt.axis('off')
    elif len(masks) == 2:
        _, axarr = plt.subplots(1,2, figsize=(10, 10))

        for i, ax in enumerate(axarr):
            ax.imshow(masks[i]["mask"], cmap="cividis")
            ax.set_title(f'{masks[i]["label"]}', {'fontsize': 15})
            ax.axis('off')
    else:
        ncols = 2
        fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(len(masks) / ncols), figsize=(15, 10))
#         for aa in axs:
#             for ax in aa:
#                 ax.axis('off')
        for i, mask in enumerate(masks):
            ax = axs[i // ncols, i % ncols]
            ax.imshow(mask["mask"], cmap="cividis")
            ax.set_title(mask["label"], {'fontsize': 15})
            ax.axis('off')
        fig.tight_layout()

    plt.show()
    
def visualize_predictions(result, save_result=False, name='result.png'):
    _, axarr = plt.subplots(2, 2, figsize=(20,10))

    axarr[0][0].imshow(result["original_image"])
    axarr[0][0].set_title('Input Image', {'fontsize': 15})
    axarr[0][0].axis('off')

    axarr[0][1].imshow(result["boxed_image"])
    axarr[0][1].set_title('Boxed Image', {'fontsize': 15})
    axarr[0][1].axis('off')

    # axarr[2].imshow(Image.open(f"../data/panoptic/{iname.split('.')[0]}.png"))
    # axarr[2].set_title('Target Mask', {'fontsize': 15})
    # axarr[2].axis('off')

    axarr[1][0].imshow(result["panoptic_mask"])
    axarr[1][0].axis('off')
    axarr[1][0].set_title('Predicted Mask', {'fontsize': 15})

    axarr[1][1].imshow(result["panoptic_image"])
    axarr[1][1].axis('off')
    axarr[1][1].set_title('Overlayed', {'fontsize': 15})

    if save_result:
        plt.savefig(f"../data/predictions/{name}", bbox_inches='tight')

    plt.show()