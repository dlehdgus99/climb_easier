import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
import utils
import argparse

predefined_colors = {

    'yellow': (0, 255, 255),
    'purple': (128, 0, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'green': (0, 255, 0),
    'blue': (77, 40, 20),
    'pink': (203, 192, 255),
}


def apply_and_generate_masks(image, bboxes):
    mask = np.zeros_like(image[:, :, 0])  # Create a mask for the whole image set to 0 (black)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2
        center_width = int(0.35 * (x2 - x1))  # 20% width of the bbox
        center_height = int(0.35 * (y2 - y1))  # 20% height of the bbox
        mask[center_y - center_height//2:center_y + center_height//2,
             center_x - center_width//2:center_x + center_width//2] = 255
    return mask


def hue_distance(h1, h2):
    """Calculate the angular distance between two hues."""
    dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
    return dh


def hsv_distance(hsv1, hsv2):
    """Calculate a perceptually more accurate distance in the HSV space."""
    hue_dist = hue_distance(hsv1[0], hsv2[0])
    sat_dist = abs(hsv1[1] - hsv2[1])
    val_dist = abs(hsv1[2] - hsv2[2])
    return np.sqrt(hue_dist**2 + sat_dist**2 + val_dist**2)


def find_dominant_color_by_clustering_hsv(image, bbox, predefined_colors_hsv, mask, num_clusters=1):
    x1, y1, x2, y2 = map(int, bbox)
    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Convert the cropped image to HSV
    hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Mask the HSV image to include only the relevant pixels
    masked_hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=crop_mask)

    # Reshape to list of pixels and filter out zero pixels (background in HSV might not be [0,0,0])
    hsv_pixels = masked_hsv_img.reshape(-1, 3)
    hsv_pixels = hsv_pixels[np.all(hsv_pixels != [0, 0, 0], axis=1)]

    if hsv_pixels.size == 0:
        print("No valid pixels under mask, skipping color calculation.")
        # Return black or a default color if no pixels are under the mask
        return (0, 0, 0)

    # Apply k-means clustering to find the dominant clusters of colors
    centroids, _ = kmeans(hsv_pixels.astype(float), num_clusters)
    cluster_labels, _ = vq(hsv_pixels, centroids)

    # Find the index of the largest cluster
    dominant_cluster_index = np.argmax(np.bincount(cluster_labels))
    dominant_color = centroids[dominant_cluster_index]

    # Find the predefined color that is closest to the dominant color
    min_distance = float('inf')
    closest_color_name = None

    for color_name, hsv_value in predefined_colors_hsv.items():
        # Compute the perceptually accurate distance from the dominant color to the predefined color
        distance = hsv_distance(dominant_color[:3], hsv_value[:3])
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name


def simplify_climb(args):
    image = cv2.imread(args.img_path)
    predefined_colors_hsv = utils.bgr_to_hsv(predefined_colors)
    detected_boxes = utils.get_all_boxes(img_path=args.img_path)
    mask = apply_and_generate_masks(image, detected_boxes)
    valid_boxes = []
    for bbox in detected_boxes:
        x1, y1, x2, y2 = bbox
        closest_color_name = find_dominant_color_by_clustering_hsv(
            image, bbox, predefined_colors_hsv, mask)
        if predefined_colors[closest_color_name] == predefined_colors[args.target_color]:
            valid_boxes.append(bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          predefined_colors[closest_color_name], 2)
            
    masked = None
    if args.show:
        cv2.imshow("Processed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        masked = utils.only_show_valid_holds(image, valid_boxes, show=True)

    if args.save:
        if masked is None:
            masked = utils.only_show_valid_holds(image, valid_boxes, show=False)

        img_name = args.img_path.split('/')[1].split('.')[0]
        cv2.imwrite(
            f'output/simplified_climb/{img_name}_{args.target_color}.png', image)
        cv2.imwrite(
            f'output/simplified_climb/{img_name}_{args.target_color}_masked.png', masked)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simplify climbing images by color.")
    parser.add_argument("--img_path", type=str,
                        default='input/color_test_4.png', help="Path to the input image")
    parser.add_argument("--target_color", type=str, choices=list(
        predefined_colors.keys()), default='yellow', help="Target color to filter")
    parser.add_argument("--show", action='store_true',
                        help="Show the processed image")
    parser.add_argument("--save", action='store_true',
                        help="Save the processed image")
    args = parser.parse_args()
    simplify_climb(args)