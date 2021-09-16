# Train DETR for object detection on custom data

In Computer Vision, object detection is a task where we want our model to distinguish the foreground objects from the background and predict the locations and the categories for the objects present in the image.

There are many frameworks out there for object detection but the researchers at Facebook AI has come up with DETR, an innovative and efficient approach to solve the object detection problem.

DETR treats an object detection problem as a direct set prediction problem with the help of an encoder-decoder architecture based on transformers. By set, I mean the set of bounding boxes. Transformers are the new breed of deep learning models that have performed outstandingly in the NLP domain. This is the first time when someone used transformers for object detection.
The authors of this paper have evaluated DETR on one of the most popular object detection datasets, COCO, against a very competitive Faster R-CNN baseline.

In the results, the DETR achieved comparable performances. More precisely, DETR demonstrates significantly better performance on large objects. However, it didn’t perform that well on small objects.

The Defacto standard to train any object detection model is to use COCO format. To train our model for object detection task we have to prepare our dataset in standard coco format

## COCO Format for Object detection

Microsoft's Common Objects in Context dataset (COCO) is the most popular object detection dataset at the moment. It is widely used to benchmark the performance of computer vision methods.

Due to the popularity of the dataset, the format that COCO uses to store annotations is often the go-to format when creating a new custom object detection dataset. While the COCO dataset also supports annotations for other tasks like segmentation, I will leave that to a future blog post. For now, we will focus only on object detection data.

The “COCO format” is a specific JSON structure dictating how labels and metadata are saved for an image dataset.

### COCO file format

If you are new to the object detection space and are tasked with creating a new object detection dataset, then following the [COCO format](https://cocodataset.org/#format-data) is a good choice due to its relative simplicity and widespread usage. This section will explain what the file and folder structure of a COCO formatted object detection dataset actually looks like.
At a high level, the COCO format defines exactly how your annotations (bounding boxes, object classes, etc) and image metadata (like height, width, image sources, etc) are stored on disk.

### Folder Structure

The folder structure of a COCO dataset looks like this:

    <dataset_dir>/
        data/
            <filename0>.<ext>
            <filename1>.<ext>
            ...
        labels.json


The dataset is stored in a directory containing your raw image data and a single json file that contains all of the annotations, metadata, categories, and other information that you could possibly want to store about your dataset. If you have multiple splits of data, they would be stored in different directories with different json files.

### JSON format

If you were to download the [COCO dataset from their website](https://cocodataset.org/#download), this would be the `instances_train2017.json` and `instances_val2017.json` files.

    {
        "info": {
            "year": "2021",
            "version": "1.0",
            "description": "Exported from FiftyOne",
            "contributor": "Voxel51",
            "url": "https://fiftyone.ai",
            "date_created": "2021-01-19T09:48:27"
        },
        "licenses": [
            {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
            },
            ...   
        ],
        "categories": [
            ...
            {
                "id": 2,
                "name": "cat",
                "supercategory": "animal"
            },
            ...
        ],
        "images": [
            {
                "id": 0,
                "license": 1,
                "file_name": "<filename0>.<ext>",
                "height": 480,
                "width": 640,
                "date_captured": null
            },
            ...
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 2,
                "bbox": [260, 177, 231, 199],
                "segmentation": [...],
                "area": 45969,
                "iscrowd": 0
            },
            ...
        ]
    }


* **Info** — Description and versioning information about your dataset.

* **Licenses** — List of licenses with unique IDs to be specified by your images.

* **Categories** — Classification categories each with a unique ID. Optionally associated with a supercategory that can span multiple classes. These categories can be whatever you want, but note that if you’d need to follow the COCO classes if you want to use a model pretrained on COCO out of the box (or follow other dataset categories to use other models).

* **Images** — List of images in your dataset and relevant metadata including unique image ID, filepath, height, width, and optional attributes like license, URL, date captured, etc.

* **Annotations** — List of annotations each with a unique ID and the image ID it relates to. This is where you will store the bounding box information in our case or segmentation/keypoint/other label information for other tasks. This also stores bounding box area and iscrowd indicating a large bounding box surrounding multiple objects of the same category which is used for evaluation.


## Creating a Custom COCO format dataset

### Background
 
If you only have unlabeled images, then you will first need to generate object labels. You can generate either ground truth labels with an annotation tool or provider (like CVAT, Labelbox, MTurk, or one of many others) or predicted labels with an existing pretrained model.

In our case we have used CVAT to create labelled dataset. The unlabeled data consists of images from construction site which contains following classes.

    [
        "aac_blocks",
        "adhesives",
        "ahus",
        "aluminium_frames_for_false_ceiling",
        "chiller",
        "concrete_mixer_machine",
        "concrete_pump",
        "control_panel",
        "cu_piping",
        "distribution_transformer",
        "dump_truck_tipper_truck",
        "emulsion_paint",
        "enamel_paint",
        "fine_aggregate",
        "fire_buckets",
        "fire_extinguishers",
        "glass_wool",
        "grader",
        "hoist",
        "hollow_concrete_blocks",
        "hot_mix_plant",
        "hydra_crane",
        "interlocked_switched_socket",
        "junction_box",
        "lime",
        "marble",
        "metal_primer",
        "pipe_fittings",
        "rcc_hume_pipes",
        "refrigerant_gas",
        "river_sand",
        "rmc_batching_plant",
        "rmu_units",
        "sanitary_fixtures",
        "skid_steer_loader",
        "smoke_detectors",
        "split_units",
        "structural_steel_channel",
        "switch_boards_and_switches",
        "texture_paint",
        "threaded_rod",
        "transit_mixer",
        "vcb_panel",
        "vitrified_tiles",
        "vrf_units",
        "water_tank",
        "wheel_loader",
        "wood_primer",
    ]

Each person was tasked to pick one class and collect images containing object for that class. These images were then uploaded to CVAT where only one class was annotated per image.

At the end of this excercise we had a number of folders each containing json file in coco format for that class annotation

    ├── aac_blocks
    │   ├── class_details.txt
    │   ├── coco.json
    │   └── images
    │       ├── img_000.png
    │       ├── img_001.png
    ├── adhesives
    │   ├── class_details.txt
    │   ├── coco.json
    │   └── images
    │       ├── img_000.png
    │       ├── img_001.png
    .
    .


As per thge standard coco format we need one `test` and one `train` file. So our next task is to combine that datset to make it one.

Before starting to create a consolidated dataset, the classes mentioned above just belongs to `things` category. But for a true panoptic segmentation we also need our dataset to include stuffs classes. In coco dataset following are the things and stuff classes.

![coco categories](./assets/cocostuff-labelhierarchy.png)

From the above image we took all low level categories and mapped them to high level categories.

    [
        {"id": 1, "name": "misc"},
        {"id": 2, "name": "textile"},
        {"id": 3, "name": "building"},
        {"id": 4, "name": "rawmaterial"},
        {"id": 5, "name": "furniture"},
        {"id": 6, "name": "floor"},
        {"id": 7, "name": "plant"},
        {"id": 8, "name": "food"},
        {"id": 9, "name": "ground"},
        {"id": 10, "name": "structural"},
        {"id": 11, "name": "water"},
        {"id": 12, "name": "wall"},
        {"id": 13, "name": "window"},
        {"id": 14, "name": "ceiling"},
        {"id": 15, "name": "sky"},
        {"id": 16, "name": "solid"},
    ]

What ever we have done till now is the most easy task, now the most important step is to find masks and bounding boxes for these mentioned categories in out custom dataset images.

### Add Masks for Stuff Classes

As we already know DETR is trained to find both stuffs and things classes. We will use pretrained DETR model to predict these low level stuff classes and map them to highlevel stuff classes.

We will use following script to do so

**Step 1:** Clone DETR Repo

    git clone https://github.com/facebookresearch/detr.git

    # Add detr folder to system path so that we can load modules from detr repo
    import sys
    sys.path.append(os.path.join(os.getcwd(), "detr/"))

**Step 2:** Load Liberaries, some of these are included from custom code we have written.

    import random
    import shutil
    import sys

    import cv2

    from categories_meta import COCO_CATEGORIES, COCO_NAMES
    from panopticapi.utils import id2rgb, rgb2id
    import panopticapi
    from PIL import Image, ImageDraw, ImageFont
    import requests
    import json
    import io
    import math
    import matplotlib.pyplot as plt
    # %config InlineBackend.figure_format = 'retina'
    import itertools

    import torch
    from torch import nn
    import torchvision.transforms as T
    import numpy as np

    torch.set_grad_enabled(False)
    # Create Original Segmented Image
    import overlay_custom_mask
    import convert_to_coco

    from categories_meta import COCO_CATEGORIES, NEW_CATEGORIES, MAPPINGS, INFO, LICENSES, cat2id, id2cat
    import coco_creator_tools

    import datetime
    import time
    import json
    import traceback


**Step 3:** Load Detr Model

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load detr model
    model, postprocessor = torch.hub.load('detr', 'detr_resnet101_panoptic', source='local', pretrained=True, return_postprocessor=True, num_classes=250)
    # Convert to eval mode
    model = model.to(device)
    model.eval()

**Step 4:** Get path to all folders containing annotations per category 

    import glob

    category_paths = glob.glob('./data/construction/*')


**Step 5:** Initialize Variables

    ROOT_DIR = './data'

    processing_file = ""
    processing_data = []

    image_id = 1
    annotation_id = 1
    segment_id = 1

    GLOBAL_COCO = {
        "licenses": LICENSES,
        "info": INFO,
        "categories": NEW_CATEGORIES,
        "annotations": [],
        "images": []
    }

    GLOBAL_PANOPTIC = {
        "licenses": LICENSES,
        "info": INFO,
        "categories": NEW_CATEGORIES,
        "annotations": [],
        "images": []
    }


**Step 6:** Run through one categoiry at a time, first coco file for that category and parse in right format. 

    ############################ Create DATASET ################################

    # run through all folders in dataset
    for category_path in category_paths:
        # store starting time
        start = time.time()
        # get category name
        category_name = category_path.split("/")[5]
        print("Processing Category:", category_name)
        # open category coco file
        with open(os.path.join(category_path, "coco.json"), "r") as coco_file:
            category_coco = json.load(coco_file)
            
        images_root = os.path.join(category_path, 'images')

**Step 7:** Create a temperary structure for images and their respective annotations

        # Process all images
        ## 1. Create a temp json which contains each image and its annotations
        ## 2. Run over this list
        ### 1. Copy this image as .jpg in GLOBAL_DIR
        ### 2. Find all segments for this image
        ### 3. Create new anotation segment which includes annotations from custom classes
        
        TEMP_COCO_IMAGES = {}
        
        # Run over all images
        for im in category_coco["images"]:
            im['annotations'] = []
            TEMP_COCO_IMAGES[im['id']] = im
            
        for ann in category_coco["annotations"]:
            TEMP_COCO_IMAGES[ann['image_id']]["annotations"].append(ann)

**Step 8:** Loop over one image at a time, as we are also going to save these images as `three channel .jpg` image we create their destinatio names

        for i, image_coco in TEMP_COCO_IMAGES.items():
            # get image path
            ## This data can be used further for logging if failed while processing
            processing_file = os.path.join(images_root, image_coco['file_name'])
            processing_data = image_coco
            output_file_name = category_name + "_" + str(image_id) + ".jpg"
            output_file_path = os.path.join(ROOT_DIR, "images", output_file_name)
            
            output_mask_name = category_name + "_" + str(image_id) + ".png"
            output_mask_path = os.path.join(ROOT_DIR, "masks", output_mask_name)

**Step 9:** Now we will read image and convert them into RGB format with 3 channels as there might be some images which are in gray format and others might be in RGBN format.

            try:

                # Read this image and get shape of image
                imo = Image.open(processing_file).convert('RGB')

                try:
                    h, w, c = np.array(imo).shape
                except:
                    h, w = np.array(imo).shape
                    c = 1

                # if no of channels != 3, open the image and convert it to 3 channel - RGB
                if c == 4 or c == 1:
                    imo = imo.convert('RGB')
                    h, w, c = np.array(imo).shape

                # Create a copy of image this will be used for further processing
                im = imo.copy()

                # Apply transform and convert image to batch
                # mean-std normalize the input image (batch-size: 1)
                img = transform(im).unsqueeze(0).to(device)  # [h, w, c] -> [1, c, ht, wt]

**Step 10:** Now we pass this formated image to model and get predicted output

                # Generate output for image
                out = model(img)

**Step 11:** Now we filter results to get all predictions above certain threshold (0.85 for us), also we pass this prediction through postprocessor to get panoptic outputs

                # Generate score
                # compute the scores, excluding the "no-object" class (the last one)
                scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]

                # threshold the confidence
                keep = scores > 0.85

                # Keep only ones above threshold
                pred_logits, pred_boxes = out["pred_logits"][keep][:, :len(
                    COCO_NAMES) - 1], out["pred_boxes"][keep]

                # the post-processor expects as input the target size of the predictions (which we set here to the image size)
                result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]






## References:

* https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4