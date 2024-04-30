from ultralytics import YOLO


IMAGE_PATH = 'input/test_image.png'
BEST_MODEL_PATH = 'trained_models/epoch_30/weights/best.pt'



from ultralytics import YOLO
import argparse


def predict_holds(img_path, model_path=BEST_MODEL_PATH, show=False, save=False):
    model = YOLO(model_path)
    results = model(img_path)

    if show:
        # Visualize the results
        for _, r in enumerate(results):
            # Show results to screen
            r.show(line_width=1, font_size=1)

    if save:
        for _, r in enumerate(results):
            img_name = args.img_path.split('/')[1].split('.')[0]
            # save results
            r.save(
                f'output/detect_holds/{img_name}_detected.png', line_width=1, font_size=1)


    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect climbing holds in an image using a trained model.")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--show", action='store_true',
                        help="Whether to show the results visually")
    parser.add_argument("--save", action='store_true',
                        help="Save the image with detected holds")
    args = parser.parse_args()

    # Run the prediction function with command-line arguments
    predict_holds(img_path=args.img_path,
                  model_path=BEST_MODEL_PATH, show=args.show, save=args.save)
