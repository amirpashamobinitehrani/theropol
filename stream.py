
import cv2
import mediapipe as mp
import math
from typing import Union, Tuple
import os
import time
import argparse
import numpy as np
import json
import torch
from network import TheroNet


#camera and screen confgis
countdown = 3
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -8.0)
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1, 0]])
SCREEN_DIMENSIONS = (1920, 1080)
pixels = (128, 128)

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def canny(image, low_thresh=100, high_thresh=200):
  edges = cv2.Canny(image, low_thresh, high_thresh)
  return edges

def is_blurry(image, thresh=150):
  edges = cv2.Laplacian(image, cv2.CV_64F)
  return edges.var() < thresh

def sharpen(image, kernel):
  sharp = cv2.filter2D(resized_down, -1, kernel)
  return sharp

def to_pixel_coords(relative_coords):
    return tuple(round(coord * dimension) for coord, dimension in zip(relative_coords, SCREEN_DIMENSIONS))


def load_model(path):
    net = TheroNet(**network_config)
    ckpt = torch.load(path)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str)
  parser.add_argument('-m', '--model_path', type=str, default='/Users/amirpashamobinitehrani/Desktop/TheroPol/MouthNet/script/exp/TheroNet/ckpt/15000.pkl')
  args = parser.parse_args()

  with open(args.config) as f:
    data = f.read()
  config = json.loads(data)

  global network_config
  network_config = config["network"]   # to define network  
  
  model = load_model(args.model_path)

  with mp_face_detection.FaceDetection(
      model_selection=0, 
      min_detection_confidence=0.5) as face_detection:
    
    for t in range(countdown):
      mins, secs = divmod(t, 60)
      timer = '{:02d}:{:02d}'.format(mins, secs)
      print(timer, end="\r")
      time.sleep(1)
      t -= 1
    
    while True:
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

          resized_down = cv2.resize(cropped_mouth, pixels, interpolation= cv2.INTER_LINEAR)
          gray_scale = cv2.cvtColor(resized_down, cv2.COLOR_BGR2GRAY)
          cv2.imshow('M', cv2.flip(gray_scale, 1)) 
          
          #conver to tensor
          gray_scale_tensor = torch.from_numpy((gray_scale/255.), ).unsqueeze(0)
          gray_scale_tensor = gray_scale_tensor.type(torch.FloatTensor).unsqueeze(0)
          
          with torch.no_grad():
            y = model(gray_scale_tensor)
            y = y.squeeze(0)
            
            if abs(y[0]) < 0.5:
              print('C')
            
            elif abs(y[1]) < 0.9:
              print('A')

            elif abs(y[2]) < 0.9:
              print('O')

            elif abs(y[3]) < 0.8:
              print('I')

            elif abs(y[4]) < 0.7:
              print('E')
            
            elif (abs(y[5])) < 1.5:
              print('U')


      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()