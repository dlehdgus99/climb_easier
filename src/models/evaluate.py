from ultralytics import YOLO
from inference import BEST_MODEL_PATH


def evaluate_model(model_weights_path=BEST_MODEL_PATH):
    model = YOLO(model_weights_path)
    metrics = model.val(data='datasets/data.yaml')  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category


if __name__ == '__main__':

    evaluate_model()

    