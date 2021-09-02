import numpy as np
import cv2
from math import floor

def get_overlayed_mask(image_size, seg):
    height, width = image_size
    # Create a blank image 
    blank_image = np.zeros((height, width))

    ## Show Code image
    # plt.imshow(image)
    # plt.show()

    # Create list of polygons to be drawn
    for i, segm in enumerate(seg["annotations"]):
        polygons_list = []
        # Add the polygon segmentation
        for segmentation_points in segm['segmentation']:
            segmentation_points = np.multiply(segmentation_points, 1).astype(int)
            polygons_list.append(segmentation_points)

        for x in polygons_list:
            end = []
            if len(x) % 2 != 0:
                print(x)
            for l in range(0, len(x), 2):
                coords = [floor(x[l]), floor(x[l + 1])]
                end.append(coords)
            contours = np.array(end)  # create a single channel 200x200 pixel black image
            if end == []:
              continue
            cv2.fillPoly(blank_image, pts=[contours], color=(1, 1, 1))
            ## Plot final image
            # plt.imshow(image)
            # plt.show()
    return blank_image