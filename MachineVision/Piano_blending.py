import cv2
import numpy as np
import matplotlib.pyplot as plt

''' A makes it more apparent d makes it less apparent '''
piano_img = cv2.imread("Large_Piano.png", cv2.IMREAD_COLOR)
dimensions = piano_img.shape
height = piano_img.shape[0]
width = piano_img.shape[1]
channels = piano_img.shape[2]

width = int(piano_img.shape[1] * 0.1)
height = int(piano_img.shape[0] * 0.1)
dim = (width, height)

piano_img = cv2.resize(piano_img, (width, height), interpolation=cv2.INTER_AREA)
def showIm(image, name="img"):
    cv2.imshow(name, image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()



# create an overlay image. You can use any image
foreground = np.ones((100, 100, 3), dtype='uint8') * 255
# Open the camera
cap= cv2.VideoCapture(0)
# Set initial value of weights
alpha = 0.4
while True:
    # read the background
    ret, background = cap.read()
    frame_h, frame_w, frame_c = background.shape

    background = cv2.flip(background, 1)
    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    added_image = cv2.addWeighted(background[0:int(frame_h/2), 0:int(frame_w), :], alpha, piano_img[0:int(frame_h/2), 0:int(frame_w), :], 1 - alpha, 0)
    # Change the region with the result
    background[0:int(frame_h/2), 0:int(frame_w)] = added_image
    # For displaying current value of alpha(weights)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(background, 'alpha:{}'.format(alpha), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('a', background)
    k = cv2.waitKey(10)
    # Press q to break
    if k == ord('q'):
        break
    # press a to increase alpha by 0.1
    if k == ord('a'):
        alpha += 0.1
        if alpha >= 1.0:
            alpha = 1.0
    # press d to decrease alpha by 0.1
    elif k == ord('d'):
        alpha -= 0.1
        if alpha <=0.0:
            alpha = 0.0
# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()