import cv2
import mediapipe as mp

import math
from typing import Union, Tuple
import os
import time
import argparse
import numpy as np

from utils import normalized_to_pixel_coordinates, canny, is_blurry, sharpen, to_pixel_coords


# image data path
data_path = '/Users/amirpashamobinitehrani/Desktop/data/train'
image_name = 'mouth'
countdown = 3

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#camera confgis
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -8.0)
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1, 0]])

SCREEN_DIMENSIONS = (1920, 1080)
count = 0


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--pose', type=int)
  parser.add_argument('-d', '--data_path', type=str, default='/Users/amirpashamobinitehrani/Desktop/data')
  parser.add_argument('-i', '--image_size', type=int, default=128)
  args = parser.parse_args()
  
  #check if folder name exists
  sub_path = os.path.join(args.data_path, str(args.pose))         
  if not os.path.exists(sub_path):
    os.mkdir(sub_path)

  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    for t in range(countdown):
      mins, secs = divmod(t, 60)
      timer = '{:02d}:{:02d}'.format(mins, secs)
      print(timer, end="\r")
      time.sleep(1)
      t -= 1
    

    while count <= 200:
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          relative_bounding_box = detection.location_data.relative_bounding_box
          
          rec_start_point = normalized_to_pixel_coordinates(
              relative_bounding_box.xmin, relative_bounding_box.ymin, image.shape[1], image.shape[0]
          )

          rect_end_point = normalized_to_pixel_coordinates(
              relative_bounding_box.xmin + relative_bounding_box.width,
              relative_bounding_box.ymin + relative_bounding_box.height,
              image.shape[1], image.shape[0]
          )
          
          xleft, ytop = rec_start_point
          xright, ybot = rect_end_point
          crop_face = image[ytop: ybot, xleft: xright]
          w, h, c = crop_face.shape
          sides = int((h / 2) / 2)
          cropped_mouth = crop_face[int(h / 2):h, 0 + sides: w - sides] 

          down_points = (args.image_size, args.image_size)
          resized_down = cv2.resize(cropped_mouth, down_points, interpolation= cv2.INTER_LINEAR)
          resized_down = cv2.cvtColor(resized_down, cv2.COLOR_BGR2GRAY)
          
          print(resized_down.shape)
          cv2.imshow('M', cv2.flip(resized_down, 1))
          image_data = os.path.join(sub_path, image_name + '_' + str(args.pose) + "_" + str(count) + '.png')
          cv2.imwrite(image_data, resized_down)
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
      count += 1
      
      print(count)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

