import face_recognition
import cv2
import numpy as np
import os, sys
import math


def face_confidence(face_distance, face_match_threshold=0.4):
    rango = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (rango * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    faces_confidences = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('No se pudo abrir la cámara')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Desconocido"
                    confidence = "Desconocido"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]: # Find a match
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        # self.face_names.append(f"{name} (confianza: {confidence})")
                        name_without_extension, _ = os.path.splitext(name)

                        self.face_names.append(name_without_extension)
                        self.faces_confidences.append(confidence)
                    else:
                        self.face_names.append(name)


            self.process_current_frame = not self.process_current_frame

            # Display annotations in the video
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Change color box depending on the name
                if name == 'Desconocido':
                    color_box = (0, 0, 255) # Red
                    acceso_text = "Acceso Denegado"
                else:
                    color_box = (0, 255, 0) # Green
                    acceso_text = "Acceso Permitido"


                cv2.rectangle(frame, (left, top), (right, bottom), color_box, 1)

                # Draw a label with the name above and confidence below the face
                #cv2.rectangle(frame, (left, bottom - 60), (right, bottom), color_box, 1)
                font = cv2.FONT_HERSHEY_DUPLEX
                name_text = f"{name}"
                confidence_text = f"{confidence}"
                cv2.putText(frame, name_text, (left + 6, bottom - 55), font, 0.6, (255, 255, 255), 1)
                if name != 'Desconocido':
                    cv2.putText(frame, confidence_text, (left + 6, bottom - 30), font, 0.6, (0, 0, 255), 1)

                cv2.putText(frame, acceso_text, (left + 6, bottom - 5), font, 0.8, (0, 0, 0), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()