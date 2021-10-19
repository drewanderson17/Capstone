
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time


# initImageforLayering:
piano_img = cv2.imread("Large_Piano.png", cv2.IMREAD_COLOR)
dimensions = piano_img.shape
height = piano_img.shape[0]
width = piano_img.shape[1]
channels = piano_img.shape[2]

width = int(piano_img.shape[1] * 0.1)
height = int(piano_img.shape[0] * 0.1)
dim = (width, height)

piano_img = cv2.resize(piano_img, (width, height), interpolation=cv2.INTER_AREA)



## For webcam input:
cap = cv2.VideoCapture(0)
alpha = 0.4
prevTime = 0
with mp_hands.Hands(
        min_detection_confidence=0.5,  # Detection Sensitivity
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, background = cap.read()
        frame_h, frame_w, frame_c = background.shape

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        background = cv2.cvtColor(cv2.flip(background, 1), cv2.COLOR_BGR2RGB)

        added_image = cv2.addWeighted(background[0:int(frame_h / 2), 0:int(frame_w), :], alpha,
                                      piano_img[0:int(frame_h / 2), 0:int(frame_w), :], 1 - alpha, 0)
        background[0:int(frame_h / 2), 0:int(frame_w)] = added_image
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        background.flags.writeable = False
        results = hands.process(background)

        # Draw the hand annotations on the image.
        background.flags.writeable = True
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    background, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(background, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow('MediaPipe Hands', background)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
