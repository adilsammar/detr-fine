import datetime
from detectron2.data import MetadataCatalog

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

NEW_CATEGORIES = [
    {"color": [51, 48, 47], "isthing": 0, "id": 1, "name": "misc"},
    {"color": [84, 140, 125], "isthing": 0, "id": 2, "name": "textile"},
    {"color": [117, 81, 69], "isthing": 0, "id": 3, "name": "building"},
    {"color": [168, 163, 135], "isthing": 0, "id": 4, "name": "rawmaterial"},
    {"color": [120, 44, 14], "isthing": 0, "id": 5, "name": "furniture"},
    {"color": [237, 235, 230], "isthing": 0, "id": 6, "name": "floor"},
    {"color": [100, 191, 31], "isthing": 0, "id": 7, "name": "plant"},
    {"color": [32, 153, 147], "isthing": 0, "id": 8, "name": "food"},
    {"color": [89, 56, 56], "isthing": 0, "id": 9, "name": "ground"},
    {"color": [87, 83, 83], "isthing": 0, "id": 10, "name": "structural"},
    {"color": [20, 178, 222], "isthing": 0, "id": 11, "name": "water"},
    {"color": [224, 75, 16], "isthing": 0, "id": 12, "name": "wall"},
    {"color": [102, 137, 145], "isthing": 0, "id": 13, "name": "window"},
    {"color": [212, 205, 205], "isthing": 0, "id": 14, "name": "ceiling"},
    {"color": [27, 104, 191], "isthing": 0, "id": 15, "name": "sky"},
    {"color": [42, 45, 48], "isthing": 0, "id": 16, "name": "solid"},
]

MAPPINGS = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    27: 1,
    28: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 1,
    41: 1,
    42: 1,
    43: 1,
    44: 1,
    46: 1,
    47: 1,
    48: 1,
    49: 1,
    50: 1,
    51: 1,
    52: 1,
    53: 1,
    54: 1,
    55: 1,
    56: 1,
    57: 1,
    58: 1,
    59: 1,
    60: 1,
    61: 1,
    62: 1,
    63: 1,
    64: 1,
    65: 1,
    67: 1,
    70: 1,
    72: 1,
    73: 1,
    74: 1,
    75: 1,
    76: 1,
    77: 1,
    78: 1,
    79: 1,
    80: 1,
    81: 1,
    82: 1,
    84: 1,
    85: 1,
    86: 1,
    87: 1,
    88: 1,
    89: 1,
    90: 1,
    92: 2,
    93: 2,
    95: 3,
    100: 4,
    107: 5,
    109: 2,
    112: 5,
    118: 6,
    119: 7,
    122: 8,
    125: 9,
    128: 3,
    130: 5,
    133: 5,
    138: 10,
    141: 2,
    144: 9,
    145: 9,
    147: 9,
    148: 11,
    149: 9,
    151: 3,
    154: 9,
    155: 11,
    156: 5,
    159: 9,
    161: 5,
    166: 3,
    168: 2,
    171: 12,
    175: 12,
    176: 12,
    177: 12,
    178: 11,
    180: 13,
    181: 13,
    184: 7,
    185: 10,
    186: 14,
    187: 15,
    188: 5,
    189: 5,
    190: 6,
    191: 9,
    192: 16,
    193: 7,
    194: 9,
    195: 4,
    196: 8,
    197: 3,
    198: 16,
    199: 12,
    200: 2,
}

# Get a list of all categories

COCO_NAMES = ["N/A"] * 201
for c in COCO_CATEGORIES:
    COCO_NAMES[c["id"]] = c["name"]

CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
    if c != "N/A":
        coco2d2[i] = count
        count += 1

# since we are treating all things as misc and that belongs to single color class, we can use colors of other things
AVAILABLE_COLORS = [
    [120, 122, 122],
    [46, 48, 99],
    [138, 142, 227],
    [133, 134, 153],
    [89, 40, 99],
    [117, 110, 76],
    [135, 140, 59],
    [109, 151, 176],
    [138, 25, 29],
    [6, 117, 71],
    [138, 116, 61],
    [181, 40, 120],
    [179, 123, 155],
    [99, 105, 125],
    [204, 18, 18],
    [138, 12, 12],
    [170, 171, 149],
    [201, 186, 12],
    [143, 9, 11],
    [122, 115, 115],
    [138, 132, 83],
    [176, 93, 46],
    [214, 210, 208],
    [128, 121, 117],
    [219, 215, 213],
    [224, 203, 193],
    [150, 72, 69],
    [145, 143, 142],
    [105, 102, 101],
    [150, 147, 90],
    [179, 152, 57],
    [168, 135, 19],
    [156, 153, 145],
    [245, 239, 223],
    [245, 194, 59],
    [102, 101, 100],
    [161, 160, 159],
    [64, 63, 62],
    [171, 174, 176],
    [45, 117, 43],
    [54, 64, 54],
    [109, 115, 109],
    [142, 163, 142],
    [206, 242, 206],
    [113, 133, 113],
    [87, 138, 148],
    [133, 130, 52],
    [204, 203, 182],
]


CUSTOM_CATEGORIES_NAMES = [
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

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/adilsammar/custom_coco_dataset",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "adilsammar",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]

category_id = 17
available_color_id = 0
for category in CUSTOM_CATEGORIES_NAMES:
    NEW_CATEGORIES.append(
        {
            "color": AVAILABLE_COLORS[available_color_id],
            "isthing": 1,
            "id": category_id,
            "name": category,
        }
    )
    category_id += 1
    available_color_id += 1

cat2id = {category["name"]: category["id"] for category in NEW_CATEGORIES}

id2cat = {i: name for name, i in cat2id.items()}


def _get_construction_instances_meta():
    thing_ids = [k["id"] for k in NEW_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in NEW_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 48, len(thing_ids)
    # Mapping from the incontiguous Construction category id to an id in [0, 47]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NEW_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_construction_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in NEW_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 16, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"]
        for k in NEW_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in NEW_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_construction_instances_meta())
    return ret


def get_builtin_metadata(dataset_name):
    if dataset_name == "construction":
        return _get_construction_instances_meta()
    if dataset_name == "construction_panoptic_separated":
        return MetadataCatalog.get("construction_panoptic_separated").set(**_get_construction_panoptic_separated_meta())

    elif dataset_name == "construction_panoptic_standard":
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in NEW_CATEGORIES]
        thing_colors = [k["color"] for k in NEW_CATEGORIES]
        stuff_classes = [k["name"] for k in NEW_CATEGORIES]
        stuff_colors = [k["color"] for k in NEW_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(NEW_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta
