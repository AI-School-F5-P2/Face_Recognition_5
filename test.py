import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os

face_objs = DeepFace.extract_faces(img_path = "data\sandra\sandra_10.jpg", detector_backend='opencv')
print(face_objs)
for face_obj in face_objs:
   img = face_obj["face"]

   plt.imshow(img)
   plt.show()