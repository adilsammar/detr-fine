import coco_creator_tools


def main(binary_mask, image_size, image_id, class_id, segmentation_id, iscrowd):

    ## --------- Prepare coco format from binary masks -------

    # {
    #     "id": 1,
    #     "image_id": 1,
    #     "category_id": 1,
    #     "segmentation": [
    #         []
    #     ],
    #     "area": 368501.0,
    #     "bbox": [
    #         0.0,
    #         74.18,
    #         751.32,
    #         544.58
    #     ],
    #     "iscrowd": 0,
    #     "attributes": {
    #         "occluded": false
    #     }
    # },

    category_info = {"id": class_id, "is_crowd": iscrowd}

    annotation_info = coco_creator_tools.create_annotation_info(
        segmentation_id, image_id, category_info, binary_mask, image_size, tolerance=2
    )

    return annotation_info
