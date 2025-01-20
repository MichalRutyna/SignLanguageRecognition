from ultralytics import YOLO

from roboflow import Roboflow

key = open("../etc/key.txt", "r").read()

rf = Roboflow(api_key=key)
project = rf.workspace("pasha-khoshkebari-ypvgn").project("asl-prediction")
version = project.version(1)
dataset = version.download("yolov11")


model = YOLO('yolo11n.pt')

#
# results = model.train(data="coco8.yaml", epochs=3)
#
# # Evaluate the model's performance on the validation set
# results = model.val()
#
# # Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")
#
# # Export the model to ONNX format
# success = model.export(format="onnx")