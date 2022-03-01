import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import pygame

pygame.init()


def DisplayText(text, image, position=(10, 450)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontcolor = (255, 255, 255)
    cv2.putText(image, text, position, font, 1, fontcolor, 2)


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, VolMax = volume.GetVolumeRange()[:2]

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []  # land mark list
    if results.multi_hand_landmarks:  # if landmarks exists
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):  # for id and lancmark in the hands
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        # mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        thumb_tip, y1 = lmList[4][1], lmList[4][2]  # thumb
        thumb_base, y2 = lmList[2][1], lmList[2][2]
        # index inger pos
        x3, middle_finger_tip = lmList[12][1], lmList[12][2]  # middle finger pos
        x3b, middle_finger_base = lmList[9][1], lmList[9][2]  # middle finger pos

        x4, ring_finger_tip = lmList[16][1], lmList[16][2]  # r finger pos
        x5, ring_finger_base = lmList[13][1], lmList[13][2]  # ring finger pos

        x6, pinkie_finger_tip = lmList[20][1], lmList[20][2]
        x7, pinkie_finger_base = lmList[17][1], lmList[17][2]

        x2b, pointer_finger_base = lmList[5][1], lmList[5][2]  # index finger Meets palm
        x2, pointer_finger_nail = lmList[8][1], lmList[8][2]  # tip of index finger

        if pointer_finger_base < pointer_finger_nail:

                pygame.mixer.Sound('assets/d1.wav').play()
                time.sleep(0.2)
                pygame.mixer.Sound('assets/d1.wav').stop()

            # DisplayText("Play note 1 on piano ", img)


        if (middle_finger_base < middle_finger_tip):
            pygame.mixer.Sound('assets/e1.wav').play()
            time.sleep(0.2)
            pygame.mixer.Sound('assets/e1.wav').stop()
            # DisplayText("Play note 2 on piano ", img)
            pass
        if (ring_finger_base < ring_finger_tip):
            pygame.mixer.Sound('assets/f1.wav').play()
            time.sleep(0.2)
            pygame.mixer.Sound('assets/f1.wav').stop()
            # DisplayText("Play note 3 on piano ", img)
            # pass
        if (pinkie_finger_base < pinkie_finger_tip):
            pygame.mixer.Sound('assets/g1.wav').play()
            time.sleep(0.2)
            pygame.mixer.Sound('assets/g1.wav').stop()
            # DisplayText("Play note 4 on piano ", img)
            pass

        if (thumb_tip < thumb_base):
            pygame.mixer.Sound('assets/c1.wav').play()
            time.sleep(0.2)
            pygame.mixer.Sound('assets/c1.wav').stop()

    

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
