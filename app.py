import face_recognition
import os
import math
import customtkinter
import threading
import tkinter as tk
from Save_data.Save_dt import ft_save_data
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys 

def face_confidence(face_distance, face_match_threshold=0.4):
    rango = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (rango * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self, root, cap):
        self.root = root
        self.cap = cap
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.faces_confidences = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True

        self.encode_faces()

        self.root.title("Face Recognition App")
        self.root.geometry("680x810")
        self.root.grid_columnconfigure(0, weight=1)

        self.camera_frame = customtkinter.CTkFrame(self.root)
        self.camera_frame.grid(row=2, column=0, pady=0)

        self.camera_label = customtkinter.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack()

        self.buttons_frame = customtkinter.CTkFrame(self.root)
        self.buttons_frame.grid(row=1, column=0, padx=20, pady=10)

        self.button1 = customtkinter.CTkButton(self.buttons_frame, text="Agregar Persona", command=self.capture_image, width=600, height=50, font=("Arial", 20))
        self.button1.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def capture_image(self):
        # Implement your code to capture and save an image
        ft_save_data(self.cap, self.root) # Replace with your actual implementation

    def run_recognition(self):
        if not self.cap.isOpened():
            sys.exit('No se pudo abrir la cámara')

        while True:
            ret, frame = self.cap.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Desconocido"
                    confidence = "Desconocido"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        name_without_extension, _ = os.path.splitext(name)

                        self.face_names.append(name_without_extension)
                        self.faces_confidences.append(confidence)
                    else:
                        self.face_names.append(name)

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == 'Desconocido':
                    color_box = (0, 0, 255)
                    acceso_text = "Acceso Denegado"
                else:
                    color_box = (0, 255, 0)
                    acceso_text = "Acceso Permitido"

                cv2.rectangle(frame, (left, top), (right, bottom), color_box, 1)

                font = cv2.FONT_HERSHEY_DUPLEX
                name_text = f"{name}"
                confidence_text = f"{confidence}"
                cv2.putText(frame, name_text, (left + 6, bottom - 55), font, 0.6, (255, 255, 255), 1)
                if name != 'Desconocido':
                    cv2.putText(frame, confidence_text, (left + 6, bottom - 30), font, 0.6, (0, 0, 255), 1)

                cv2.putText(frame, acceso_text, (left + 6, bottom - 5), font, 0.8, (0, 0, 0), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

            self.root.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    root = customtkinter.CTk()
    cap = cv2.VideoCapture(0)
    app = FaceRecognition(root, cap)
    face_thread = threading.Thread(target=app.run_recognition)
    face_thread.daemon = True
    face_thread.start()

    root.mainloop()