import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLO


def main():
    model = YOLO("cfg/models/mlf/yolo-mlfx.yaml")
    data_config = "cfg/datasets/coco128.yaml"

    model.train(data=data_config, epochs=500, batch=16, imgsz=640)
    metrics = model.val()
    path = model.export(format="onnx")

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()
