!pip install ultralytics

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


video_path = "/content/154647-808044372_small.mp4"
video = cv2.VideoCapture(video_path)

model = YOLO("yolov8s.pt")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 20.0,
                      (int(video.get(3)), int(video.get(4))))


while video.isOpened():
  return_value, frame = video.read()
  if return_value == False:
    break #stops if video ends

  #performs object detection
  result = model(frame)

  bounded_frame = result[0].plot() #get the frame with bounding boxes

  out.write(bounded_frame)


video.release()
out.release()
print("Video saved as output.mp4")
