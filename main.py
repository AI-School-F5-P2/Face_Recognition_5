import threading
import cv2
from deepface import DeepFace
import os
import time


def check_face(frame):
    global recognized_person
    global data

    for person_name, photos in data.items():
        for photo_path in photos:
            try:
                reference_image = cv2.imread(photo_path)
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

                result = DeepFace.verify(frame, reference_image.copy(), distance_metric = "cosine")

                if result['verified']:
                    recognized_person = person_name
                    confidence = 1 - result['distance']
                    return recognized_person, confidence # Exit the function when a face is recognized
            except ValueError:
                pass

    recognized_person = "Desconocido"
    confidence = 0
    return recognized_person, confidence

def count_photos(dir, extension):
    try:
        # list all files in the directory
        files = os.listdir(dir)

        # Filter the files with the extension specified
        files_with_ext = [file for file in files if file.endswith(extension)]

        # Count the number of files with the extension specified
        total_files = len(files_with_ext)

        return total_files

    except Exception as e:
        print(f"Error al contar archivos: {str(e)}")
        return None


# Max number of photos to take
max_photos = 31


# load Haarcascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_eye.xml')

counter = 0

# Directorio base para guardar las fotos
output_base_dir = "data"
recognized_person = 'Desconocido'

# load photos route for each person
data = {}
root_dir = 'data'

for person_name in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_name)

    if os.path.isdir(person_dir):
        photos = [os.path.join(person_dir, photo) for photo in os.listdir(person_dir) if photo.endswith('.jpg')]
        data[person_name] = photos

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
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors = 5, minSize=(30, 30))
        except Exception as e:
            print(f"Error in face detection: {e}")
        # draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # Create a rectangle around the faces
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

            # Check if the face is recognized
            if counter % 30 == 0:
                try:
                    recognized_person, confidence = check_face(frame[y:y + h, x:x + w].copy())
                    threading.Thread(target=check_face, args=(frame[y:y+h, x:x+w].copy(),)).start()
                except ValueError:
                    pass
            counter += 1

            if recognized_person != 'Desconocido':
                saved_photos = count_photos(os.path.join(output_base_dir, recognized_person), '.jpg')
                # save the ROI
                roi = frame[y:y + h, x:x + w]
                # create a file with the ROI
                if saved_photos < max_photos:
                    timestamp = int(time.time())
                    unique_identifier = f"{recognized_person.upper()}_{timestamp}"
                    output_dir = os.path.join(output_base_dir, recognized_person)
                    # Verify if the directory exists and create it if not
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, f'{unique_identifier}_roi_{saved_photos}.jpg')
                    cv2.imwrite(output_path, roi)
                    saved_photos += 1

                # draw a rectangle around the faces with border green
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (72, 131, 72), 3)
                cv2.putText(
                    frame,
                    f"{recognized_person.upper()}",
                    (x, y + h + 40),  # display name below the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),  # text color
                    2
                )

                cv2.putText(
                    frame,
                    "ACCESO PERMITIDO",
                    (x, y + h + 75),  # display below the name
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (72, 131, 72),  # text color
                    2
                )

                cv2.putText(
                    frame,
                    f"Seguridad: {confidence:.2%}",
                    (x, y + h + 110),  # display below Access Allowed
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),  # text color
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"{recognized_person.upper()}",
                    (x, y + h + 40),  # display 'Desconocido' below the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),  # text color
                    2
                )

                cv2.putText(
                    frame,
                    "ACCESO DENEGADO",
                    (x, y + h + 75),  # Display below 'Desconocido'
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),  # text color
                    2
                )

        cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()