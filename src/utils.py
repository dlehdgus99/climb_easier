import cv2
from models import inference
import numpy as np 

def bgr_to_hsv(bgr_colors_dict):
    predefined_colors_hsv = {}
    for color_name, bgr_value in bgr_colors_dict.items():
        bgr_array = np.uint8([[bgr_value]])
        hsv_value = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
        predefined_colors_hsv[color_name] = hsv_value
    return predefined_colors_hsv


def get_all_boxes(img_path,show=True):
    results = inference.predict_holds(img_path=img_path, show=show)
    bbox_list = results[0].boxes.xyxy.tolist()
    int_bbox_list = [[int(item) for item in sublist] for sublist in bbox_list]
    return int_bbox_list


def only_show_valid_holds(img, valid_boxes, show=True):

    # Initialize a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(img)

    # Draw rectangles on the mask for each target hold
    for (x1, y1, x2, y2) in valid_boxes:
        mask[y1:y2, x1: x2] = img[y1: y2, x1:x2]

    # Optionally, set the non-target areas to a specific color (e.g., white)
    # You can skip this step if you only want the target holds on a black background
    non_target_color = [60, 60, 60]  # grey
    mask[np.where((mask == [0,0,0]).all(axis=2))] = non_target_color

    # Display the result
    if show:
        cv2.imshow('Target Holds Only', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        

    return mask
