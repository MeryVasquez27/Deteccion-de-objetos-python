import torch
import cv2
import numpy as np

#Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'C:\Users\jgpac\Desktop\Project_Analisis')

#Realizar videocaptura
cap = cv2.VideoCapture(1)

#Inicio 
while True:
    #Realizar lectura de la videocaptura
    ret, frame = cap.read()

    #Realizar deteccion
    detect = model(frame)

    info = detect.pandas().xyxy[0]
    print(info)

    #Mostramos FPS
    cv2.imshow('Detector de Objetos', np.squeeze(detect.render()))

    #Leer teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()

