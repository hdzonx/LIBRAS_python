'''' Código python para compreender Lingua Brasileira de Sinais
    usando OpenCV, conforme o tutorial do Murtaza's Workshop em
    https://www.youtube.com/watch?v=wa2ARoUUdU8.'''

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import  Classifier

import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Data/Model/keras_model.h5", "Data/Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands = detector.findHands(img, draw = False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    #cv2.imshow("Image", img)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    #É necessário clicar na caixa de diálogo da imagem chamada ImageWhite
    #if key == ord("s"): # se a tecla pressionada for s, faça:
        #counter += 1
        #cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) #Salva no folder um screnshot da imagem cada vez que se pressiona s
        #print("pressed s")
        #print(counter)
