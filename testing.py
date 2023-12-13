import cv2
import mediapipe as mp
import numpy as np
import math
import os

nombre = "rostros"
direccion = "C:/Users/Usuario/Desktop/ProyectoFinal/ProyectoFinal/ProyectoFinal/rostros"
carpeta = direccion + "/" + nombre
if not os.path.exists(carpeta):
    print("Carpeta creada: ", carpeta)
    os.makedirs(carpeta)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detros = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
dibujorostro = mp.solutions.drawing_utils

def mouse(evento, xm, ym, bandera, param):
    global xmo, ymo, marca

    if evento == cv2.EVENT_LBUTTONDOWN:
        xmo = xm
        ymo = ym
        marca = 1

marca = 0
xp, yp = 0, 0
xa, ya = 0, 0
sua = 5
pubix, pubiy = 0, 0
listax = []
listay = []
circulos = []
con = 0
por = 0
puntos = []

cv2.namedWindow("Face Detection")
cv2.setMouseCallback("Face Detection", mouse)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    copia = frame.copy()
    copia2 = frame.copy()

    RGB = cv2.cvtColor(copia, cv2.COLOR_BGR2RGB)

    if marca == 0:
        interfaz = cv2.imread("./hola.jpg")
      
        if interfaz is not None:
            interfaz_resized = cv2.resize(interfaz, (frame.shape[1], frame.shape[0]))

            if interfaz_resized.shape[2] == frame.shape[2]:
                frame[0:720, 0:1280] = interfaz_resized
                cv2.imshow("Face Detection", frame)
            else:
                print("Número incorrecto de canales en la imagen de interfaz.")
        else:
            print("Error al leer la interfaz.")
    
    elif marca == 1:
        interfaz2 = cv2.imread("./adios.jpg")
        height = frame.shape[0]
        width = frame.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.array([[[628, 49], [824, 130], [986, 326], [824, 525], [628, 606], [432, 525], [270, 326], [432, 130]]], dtype=np.int32)

        cv2.fillPoly(mask, pts, 255)

        zona = cv2.bitwise_and(frame, frame, mask=mask)
        zona_resized = cv2.resize(zona, (interfaz2.shape[1], interfaz2.shape[0]))

        interfaz2[zona_resized > 0] = 0

        resrostros = detros.process(interfaz2)

        if resrostros.detections is not None:
            for rostro in resrostros.detections:
                dibujorostro.draw_detection(interfaz2, rostro)
                x1 = int(rostro.location_data.relative_keypoints[2].x * width)
                y1 = int(rostro.location_data.relative_keypoints[2].y * height)
                
                x1 = int(pubix + (x1 - pubix) / sua)
                y1 = int(pubiy + (y1 - pubiy) / sua)

                xci = int(rostro.location_data.relative_bounding_box.xmin * width)
                yci = int(rostro.location_data.relative_bounding_box.ymin * height)

                ancho = int(rostro.location_data.relative_bounding_box.width * width)
                alto = int(rostro.location_data.relative_bounding_box.height * height)
                xf, yf = xci + ancho, yci + alto

                x2 = (xci + xf) // 2
                y2 = (yci + yf) // 2

                x3 = x2 + 50
                y3 = y2 

                cv2.line(interfaz2, (x1, y1), (x2, y2), (128, 0, 128), 1)
                cv2.line(interfaz2, (x2, y2), (x3, y3), (255, 255, 255), 1)
                cv2.line(interfaz2, (x3, y3), (x1, y1), (255, 255, 0), 1)
                
                cv2.circle(interfaz2, (x1, y1), 2, (255, 255, 255), 2)
                cv2.circle(interfaz2, (x2, y2), 2, (255, 255, 0), 2)
                cv2.circle(interfaz2, (x3, y3), 2, (128, 0, 128), 2)
                
                r = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                theta = math.atan2(y1 - y2, x1 - x2) / math.pi * 180
                theta = int(theta)

                pubix, pubiy = x1, y1

                if theta < 0 and por <= 100:
                    theta = theta + 360

                cv2.putText(interfaz2, str(theta), (148, 695), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(interfaz2, str(por) + "%", (1160, 695), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                pruebax, pruebay = cv2.polarToCart(290, theta, x=627, y=327, angleInDegrees=True)
                dibx, diby = int(pruebax[0] * -1) + 627, int(pruebay[0] * -1) + 327

                imgristros = copia2[yci:yf, xci:xf]
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(con), imgristros)
        
                circulos.append((dibx, diby))
                listax.insert(con, dibx)
                listay.insert(con, diby)
                print(len(circulos))

                for cir in circulos:
                    rx, ry = cir[0], cir[1]

                    cv2.circle(interfaz2, (rx, ry), 4, (255, 255, 0), -1)

                    puntos = len(circulos)
                    if len(circulos) >= 480:
                        cv2.putText(interfaz2, "COMPLETADO", (500, 695), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                con = con + 1

            por = int((puntos * 100) / 480)

            cv2.imshow("Face Detection", interfaz2)

    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
