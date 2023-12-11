import customtkinter
import cv2
import threading
from PIL import Image, ImageTk
from deepface import DeepFace
import os
import tkinter as tk
from Save_data.Save_dt import ft_save_data
from tkinter import PhotoImage



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
recognized_person = "Desconocido"
# load photos route for each person
data = {}
root_dir = 'data'

for person_name in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_name)

    if os.path.isdir(person_dir):
        photos = [os.path.join(person_dir, photo) for photo in os.listdir(person_dir) if photo.endswith('.jpg')]
        data[person_name] = photos



# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Configurar la interfaz gráfica
app = customtkinter.CTk()
app.title("Face Recognition")
app.geometry("680x600")
app.grid_columnconfigure((0), weight=1)


# Frame para la cámara
camera_frame = customtkinter.CTkFrame(app)
camera_frame.grid(row=2, column=0, padx=20, pady=0)

# Etiqueta para mostrar la cámara
camera_label = customtkinter.CTkLabel(camera_frame)
camera_label.pack()

# Frame para los botones
buttons_frame = customtkinter.CTkFrame(app)
buttons_frame.grid(row=1, column=0, padx=20, pady=20)


# Botón 1
button1 = customtkinter.CTkButton(buttons_frame, text="Registrate",command=lambda: ft_save_data(cap, app) ,width=600, height=50, font=("Arial", 20))
button1.grid(row=0, column=0, padx=20, pady=20, sticky="ew")


def update_camera():
    global counter
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
                    threading.Thread(target=check_face, args=(frame[y:y+h, x:x+w].copy(),)).start()
                except ValueError:
                    pass
            counter += 1

            if recognized_person != 'Desconocido':
                cv2.putText(frame, f"Persona: {recognized_person}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Convierte la imagen para mostrar en customtkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            camera_label.configure(image=photo)
            camera_label.image = photo

        # Vuelve a llamar a la función después de 10 ms
        app.after(10, update_camera)

# Iniciar la actualización de la cámara
update_camera()

# Iniciar la interfaz gráfica
app.mainloop()

# Liberar la cámara cuando se cierra la ventana
cap.release()
