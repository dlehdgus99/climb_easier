from ultralytics import YOLO


def train_model(model_weights_path=None,epochs=5,resume=False):
    model = None
    if model_weights_path:
        print(f'resuming training for {model_weights_path}')
        model = YOLO(model_weights_path)
        model.train(data='datasets/data.yaml', epochs=epochs,
                    device='mps', batch=16, imgsz=640, save=True, plots=True,resume=resume)

    else:
        model = YOLO('yolov8n.pt')
        model.train(data='datasets/data.yaml', epochs=epochs,
                    device='mps', batch=16, imgsz=640, save=True, plots=True)
        



if __name__ == '__main__':
    model_weights_path = None
    train_model(model_weights_path=model_weights_path, epochs=30, resume=True)