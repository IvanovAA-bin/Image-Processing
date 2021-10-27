import cv2 as cv
import numpy as np
#import os

image_base = "./ListsDataset/"
filters_base = "./GeneratedFilters/"
mouse_x = 0
mouse_y = 0
img = None
hsv_img = None
chosen = None
ill_part = None
img_number = 0
current_img_number = -1
img_max_number = 5000
pixels_set = []

win_source_name = "source image"
win_chosen_name = "chosen px"
win_inRange_name = "inRange"

l_H = l_S = l_V = 255
h_H = h_S = h_V = 0


def resetHSVparams():
    global l_H, l_S, l_V, h_H, h_S, h_V
    l_H = 160
    l_S = l_V = 255
    h_H = h_S = h_V = 0


def clbk(event, x, y, flags, param):
    global mouse_y, mouse_x, img, hsv_img
    global l_H, l_S, l_V, h_H, h_S, h_V
    mouse_x = x
    mouse_y = y
    if flags == 1:
        chosen[y, x] = (255, 0, 255)
        l_H = int(min(l_H, hsv_img[y, x, 0]))
        h_H = int(max(h_H, hsv_img[y, x, 0]))
        l_S = int(min(l_S, hsv_img[y, x, 1]))
        h_S = int(max(h_S, hsv_img[y, x, 1]))
        l_V = int(min(l_V, hsv_img[y, x, 2]))
        h_V = int(max(h_V, hsv_img[y, x, 2]))

    # if (event == cv.EVENT_LBUTTONDBLCLK):
    #   cv.circle(img, (x, y), 20, (255, 0, 255), -1)
    # print(event, x, y, flags, param)


resetHSVparams()

cv.namedWindow(win_source_name)
cv.namedWindow(win_inRange_name)
cv.namedWindow(win_chosen_name)
cv.setMouseCallback(win_source_name, clbk)

while True:
    if current_img_number != img_number:
        current_img_number = img_number
        img = cv.imread(image_base + str(img_number) + ".jpg")
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        chosen = np.zeros_like(img, np.uint8)

    if l_H < h_H and l_S < h_S and l_V < h_V:
        ill_part = cv.inRange(hsv_img, (l_H, l_S, l_V), (h_H, h_S, h_V))
        cv.imshow(win_inRange_name, ill_part)

    cv.imshow(win_source_name, img)
    cv.imshow(win_chosen_name, chosen)
    k = cv.waitKey(20) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif k == ord('c'):
        resetHSVparams()
        cv.imshow(win_inRange_name, np.zeros_like(img, np.uint8))
        chosen = np.zeros_like(img, np.uint8)
    elif k == ord('a'):
        img_number -= 1
        img_number = max(0, img_number)
    elif k == ord('d'):
        img_number += 1
        img_number = min(img_max_number, img_number)
    elif k == ord('s'):
        f = open(str(img_number)+".txt", 'w')
        f.write(str(l_H) + " " + str(l_S) + " " + str(l_V) + " " + str(h_H) + " " + str(h_S) + " " + str(h_V))
        f.close()



