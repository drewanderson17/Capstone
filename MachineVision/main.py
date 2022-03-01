import math

import mediapipe as mp
import cv2
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

#Piano Parameters
white = (255, 255, 255)
green = (0, 255, 0)
topY = 60
botY = 250
Width = 70
startLeft = 30
numKeys = 16
keyDisplacement = 2
keys = list()

#global bits
volumeControlBit = 0

class PianoKey:
  def __init__(self, topLeftXY, botRightXY):
      self.topLeftXY = topLeftXY
      self.botRightXY = botRightXY



def init_piano():
    temp = startLeft
    for i in range(numKeys):
        keys.append(PianoKey((temp, topY), (temp + Width, botY)))
        temp += Width + keyDisplacement

def draw_piano(image, keyPress):
    for i in range(len(keys)):
        if keyPress == i:
            cv2.rectangle(image, keys[i].topLeftXY, keys[i].botRightXY, green, cv2.FILLED)
        else:
            cv2.rectangle(image, keys[i].topLeftXY, keys[i].botRightXY, white, cv2.FILLED)

def activate_volume_control(vol_vec):
    x1, y1, x2, y2 = vol_vec
    h, w, c = image.shape
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    length = math.hypot(x2 - x1, y2 - y1)
    draw_volume_bar(length)

def draw_volume_bar(length):
    # org
    org = (1050, 100)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    cv2.putText(image, 'Volume', org, font,
                      fontScale, color, thickness, cv2.LINE_AA)

    volBar, volPercent = interpret_volumes(length)
    cv2.rectangle(image, (1100, 150), (1135, 400), (0, 0, 0), 3)
    cv2.rectangle(image, (1100, 150), (1135, 400), (0, 0, 0), 3)
    cv2.rectangle(image, (1100, int(volBar)), (1135, 400), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, f'{int(volPercent)} %', (1100, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 3)

def interpret_volumes(length):
    volBar = np.interp(length, [50, 220], [400, 150])
    volPercent = np.interp(length, [50, 220], [0, 100])
    return (volBar, volPercent)


#Needs position of index finger
def check_in_piano_range(x_pos, y_pos, f_vec):
    for i in range(numKeys):
        if x_pos > keys[i].topLeftXY[0] and x_pos < keys[i].botRightXY[0] and y_pos > keys[i].topLeftXY[1] and y_pos < keys[i].botRightXY[1] and pressingGesture(f_vec):
            print(i)
            return i
    return -1

def dist(p1x,p2x,p1y,p2y):
    distance = math.sqrt((math.pow(p2y-p1y, 2))+(math.pow(p2x-p1x, 2)))
    return distance * 100

def pressingGesture(arr):
    tx,ty,dx,dy,px,py=arr
    dis1 = dist(tx,dx,ty,dy)
    dis2 = dist(tx,px,ty,py)
    dis3 = dist(dx,px,dy,py)
    if(dis1 < 3.4 and dis2 < 5 and dis3 < 4):
        return True
    return False

def pressingGesture2(arr):
    tx,ty,bx,by = arr
    dis1 = dist(tx,bx,ty,by)
    if(dis1 < 3.4):
        return True
    return False

def pinchingGesture(arr):
    thumbTipX, thumbTipY, indexTipX, indexTipY = arr
    distance = dist(thumbTipX, indexTipX, thumbTipY, indexTipY)
    if (distance < 4):
        return True

#setup holistic model
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
    #Get capture device
    cap = cv2.VideoCapture(1)

    image_width = cap.get(3)
    image_hight = cap.get(4)

    init_piano()

    while cap.isOpened():
        #read feed
        ret, frame = cap.read()

        #recolor frame, holistic model accepts rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)


        results = hands.process(image)
        #convert back to BGR for opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        key = -1
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_pos_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                y_pos_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                x_pos_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
                y_pos_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                x_pos_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                y_pos_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                x_pos_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                y_pos_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

                x_thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                y_thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                vol_vec = [x_thumb_tip, y_thumb_tip, x_pos_tip, y_pos_tip]

                if volumeControlBit == 1:
                    activate_volume_control(vol_vec)


                f_vec = [x_pos_tip,y_pos_tip,x_pos_dip,y_pos_dip,x_pos_pip,y_pos_pip]

                x_pos = x_pos_tip * image_width
                y_pos = y_pos_tip * image_hight
                key = check_in_piano_range(x_pos, y_pos, f_vec)

        #render results
        blk = np.zeros(image.shape, np.uint8)
        draw_piano(blk, key)
        out = cv2.addWeighted(image, 1.0, blk, 0.5, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX



        cv2.imshow("Airpiano", out)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Quit")
            break

        if cv2.waitKey(10) & 0xFF == ord('v'):
            if volumeControlBit == 0:
                volumeControlBit = 1
                print("activate volume control")
            else:
                print("deactivate volume control")
                volumeControlBit = 0

    cap.release()
    cv2.destroyAllWindows



