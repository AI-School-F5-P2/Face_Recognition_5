import customtkinter
import cv2
import threading
from PIL import Image, ImageTk
from Button import button_callback

def update_camera():
    # Función para actualizar la vista de la cámara
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(frame)
    camera_label.configure(image=photo)
    camera_label.image = photo
    camera_label.after(10, update_camera)  # Llama a la función después de 10 ms

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Configurar la interfaz gráfica
app = customtkinter.CTk()
app.title("Mi Aplicación")
app.geometry("680x600")



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
button1 = customtkinter.CTkButton(buttons_frame, text="Tomar foto!", command=lambda: button_callback(cap), width=200, height=50, font=("Arial", 20))
button1.grid(row=0, column=0, padx=10, pady=10)

# Botón 2
button2 = customtkinter.CTkButton(buttons_frame, text="Escanear", command=button_callback, width=200, height=50, font=("Arial", 20))
button2.grid(row=0, column=1, padx=10, pady=10)

# Iniciar la actualización de la cámara en un hilo separado
camera_thread = threading.Thread(target=update_camera)
camera_thread.start()

# Iniciar la interfaz gráfica
app.mainloop()

# Liberar la cámara cuando
