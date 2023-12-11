import customtkinter
import cv2
import threading
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import Toplevel, Label, Entry, Button

# Variable global para la referencia persistente a la imagen
imagen_tk = None

def mostrar_foto(ruta_imagen):
    # Función para mostrar la foto en una ventana emergente
    popup = Toplevel()
    popup.title("Foto Capturada")

    # Cargar la imagen desde la ruta proporcionada
    print(ruta_imagen)
    imagen = Image.open("./" + ruta_imagen)

    # Mostrar la imagen en una etiqueta
    etiqueta_imagen = Label(popup, image=ImageTk.PhotoImage(imagen))
    etiqueta_imagen.photo = ImageTk.PhotoImage(imagen)
    etiqueta_imagen.pack()

    # Botón para cerrar la ventana emergente
    boton_cerrar = Button(popup, text="Cerrar", command=popup.destroy)
    boton_cerrar.pack()

    popup.mainloop()


def button_callback(cap):
    # Crear la carpeta "images" si no existe
    if not os.path.exists("images"):
        os.makedirs("images")

    # Función que se ejecuta al presionar el botón
    _, frame = cap.read()

    # Ventana para ingresar el nombre
    nombre_popup = Toplevel()
    nombre_popup.title("Ingresar Nombre")

    # Etiqueta y entrada para el nombre
    nombre_label = Label(nombre_popup, text="Ingrese su nombre:")
    nombre_label.pack()

    nombre_entry = Entry(nombre_popup)
    nombre_entry.pack()

    def guardar_con_nombre():
        nombre = nombre_entry.get()
        if nombre:
            # Guardar la imagen con el nombre proporcionado
            ruta_guardado = f"images/{nombre}_captura.jpg"
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(ruta_guardado)

            # Cerrar la ventana
            nombre_popup.destroy()

            # Mostrar la foto
            mostrar_foto(ruta_guardado)

    # Botón para guardar con nombre
    boton_guardar = Button(nombre_popup, text="Guardar", command=guardar_con_nombre)
    boton_guardar.pack()

    # Mantener la ventana abierta
    nombre_popup.mainloop()