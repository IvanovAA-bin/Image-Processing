import cv2 as cv
import numpy as np
import os


def DFFTnp(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def reverceDFFTnP(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverce_image = np.fft.ifft2(f_ishift)
    return reverce_image


def clbk(event, x, y, flags, param):
    global dfft_img_to_show
    global dfft_img_preshow
    global dfft_img
    dfft_img_to_show = np.copy(dfft_img_preshow)
    Y = dfft_img_to_show.shape[0] - y
    X = dfft_img_to_show.shape[1] - x
    dfft_img_to_show[y:y+10, x:x+10] = 0
    dfft_img_to_show[Y - 10:Y, X - 10:X] = 0

    if flags == 1:
        dfft_img[y:y+10, x:x+10] = 0
        dfft_img[Y - 10:Y, X - 10:X] = 0
        dfft_img_preshow = np.log2(abs(dfft_img + 1))
        dfft_img_preshow = cv.normalize(dfft_img_preshow, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                        dtype=cv.CV_64F)

    #print(event, x, y, flags, param)


img_number = 0
current_img_number = -1
win_src_name = "source image"
win_fft_name = "dfft image"
win_cvt_name = "converted image"


os.chdir("./imgs")
imlist = os.listdir(".")
img_max_number = len(imlist)

cv.namedWindow(win_src_name)
cv.namedWindow(win_fft_name)
cv.namedWindow(win_cvt_name)
cv.setMouseCallback(win_fft_name, clbk)

print(imlist)

img = None
dfft_img = None
dfft_img_to_show = None
dfft_img_preshow = None
while True:
    if current_img_number != img_number:
        current_img_number = img_number
        img = np.float32(cv.imread(imlist[img_number], 0))
        img = cv.resize(img, (600, 400), cv.INTER_AREA)
        dfft_img = DFFTnp(img)
        dfft_img_preshow = np.log2(abs(dfft_img + 1))
        dfft_img_preshow = cv.normalize(dfft_img_preshow, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_64F)
        dfft_img_to_show = np.copy(dfft_img_preshow)

    cv.imshow(win_src_name, np.uint8(img))
    cv.imshow(win_fft_name, np.uint8(dfft_img_to_show))
    cv.imshow(win_cvt_name, np.uint8(reverceDFFTnP(dfft_img)))

    k = cv.waitKey(20) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif k == ord('a'):
        img_number -= 1
        img_number = max(0, img_number)
    elif k == ord('d'):
        img_number += 1
        img_number = min(img_max_number - 1, img_number)
















