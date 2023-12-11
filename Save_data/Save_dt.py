import cv2
from PIL import Image, ImageTk
import os
import customtkinter as ctk

# Variable global para la referencia persistente a la imagen
imagen_tk = None


def ft_save_data(cap, app):
    _, frame = cap.read()

    # Crear una carpeta temporal
    temp_folder = "./data/temp_folder"
    os.makedirs(temp_folder, exist_ok=True)

    # Guardar la captura en la carpeta temporal
    image_path = os.path.join(temp_folder, "temp_capture.jpg")
    cv2.imwrite(image_path, frame)

    # Crear una nueva ventana para mostrar la imagen capturada
    save_data_window = ctk.CTkToplevel(app)
    save_data_window.title("Guardar Foto")

    # Cargar la imagen capturada
    captured_image = Image.open(image_path)
    photo = ImageTk.PhotoImage(captured_image)

    # Mostrar la imagen en la nueva ventana
    image_label = ctk.CTkLabel(save_data_window, image=photo)
    image_label.image = photo
    image_label.pack()

    # Label para el nombre de la foto
    name_label = ctk.CTkLabel(save_data_window, text="Ingresa nombre y apellido", width=600, height=50, font=("Arial", 20))
    name_label.pack()

    # Entry para ingresar el nombre de la foto
    name_entry = ctk.CTkEntry(save_data_window,width=600, height=50, font=("Arial", 20))
    name_entry.pack()

    def save_photo():
        # Obtener el nombre ingresado por el usuario
        photo_name = name_entry.get()

        # Guardar la imagen con el nombre proporcionado en la carpeta temporal
        save_path = os.path.join(temp_folder, f"{photo_name}.jpg")
        captured_image.save(save_path)

        # Cambiar el nombre de la carpeta temporal al nombre de la imagen
        new_folder_name = os.path.join("./data", photo_name)
        os.rename(temp_folder, new_folder_name)

       

        # Cerrar la ventana después de guardar la foto
        save_data_window.destroy()
        app.destroy()
        # volver a la ventana principal
        # llamar a la funcion principal en app.py

        

    # Botón para guardar la foto
    save_button = ctk.CTkButton(save_data_window, text="Guardar", command=save_photo, width=600, height=50, font=("Arial", 20))
    save_button.pack()