import threading
import cv2
from deepface import DeepFace
import os


def check_face(frame):
    global recognized_person
    global data

    for person_name, photos in data.items():
        for photo_path in photos:
            try:
                reference_image = cv2.imread(photo_path)
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

                if DeepFace.verify(frame, reference_image.copy())['verified']:
                    recognized_person = person_name
                    return  # Sale del bucle cuando encuentra una coincidencia
            except ValueError:
                pass

    recognized_person = "Desconocido"

# load Haarcascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


counter = 0
recognized_person = 'Desconocido'

# load photos route for each person
data = {}
root_dir = 'data'

for person_name in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_name)

    if os.path.isdir(person_dir):
        photos = [os.path.join(person_dir, photo) for photo in os.listdir(person_dir) if photo.endswith('.jpg')]
        data[person_name] = photos



#reference_image = cv2.imread('photos/sandra.jpg')
#reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Establece el tamaño de la ventana manualmente
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',1280, 720)

while True:
    ret, frame = cap.read()
    if ret:
        # convert to gray scale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces with the model Haarcascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # draw a rectangle around the faces
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        if counter % 60 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if recognized_person != 'Desconocido':
            cv2.putText(
                frame, f"Persona: {recognized_person}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()