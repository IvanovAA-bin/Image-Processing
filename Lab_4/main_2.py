# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:10:49 2021

@author: user
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sg
import math


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Gabor Filter
def Gabor_filter(K_size = 111, Sigma = 10, Gamma = 1.2, Lambda = 10, Psi = 0, angle = 0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Используйте фильтр Габора, чтобы воздействовать на изображение
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    # gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
    #plt.imshow(gabor)
    #plt.show()

    # filtering

    out = cv2.filter2D(gray, -1, gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Используйте 6 фильтров Габора с разными углами для извлечения деталей на изображении
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)
    gray = 255 - gray[...]

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        #_out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)
        _out = Gabor_filtering(gray, K_size=15, Sigma=5, Gamma=1.2, Lambda=10, angle=A)
        #plt.imshow(_out, cmap='gray')
        #plt.show()

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


def mainAngle(imPart, kernelSize, angles, diff=0.1, sigma=15, gamma=1.8, Lambda=10, psi=0):
    values = np.zeros(len(angles))
    i = int(0)
    maxV = int(0)
    indMax = int(0)
    for angle in angles:
        gabor = Gabor_filter(K_size=kernelSize, Sigma=sigma, Gamma=gamma, Lambda=Lambda, Psi=psi, angle=angle)
        filtered = sg.convolve2d(imPart, gabor, mode="valid")
        val = filtered[0, 0]
        #print("val =", val)
        values[i] = val
        if i == 0:
            maxV = val
            indMax = 0
        else:
            if val > maxV:
                maxV = val
                indMax = i
        i += 1
    #print("max =", maxV, "\tangle =", angles[indMax])
    #print("filtered = ", values)
    #print("average =", np.average(values))
    if abs(maxV - np.average(values)) < diff:
        return -1
    else:
        return angles[indMax]


def turnMask(img, kernelSize, anglesCount, sigma=15, gamma=1.8, Lambda=10, psi=0):
    img = BGR2GRAY(img)
    img = 255 - img[...]
    if 180 % anglesCount != 0:
        raise Exception("360 % anglesCount must be equal 0")
    h_im, w_im = img.shape
    if h_im != w_im:
        raise Exception("shapes of img must be square")
    if h_im % kernelSize != 0:
        raise Exception("image side % kernel side must be equal 0")
    angles = np.arange(0, 180, 180 / anglesCount)
    n = h_im // kernelSize
    mask = np.zeros((n, n), dtype=np.int16)
    for i in range(0, n, 1):
        print("processing line", i + 1, "of", n)
        for j in range(0, n, 1):
            subImg = img[i * kernelSize: (i + 1) * kernelSize, j * kernelSize: (j + 1) * kernelSize]
            #cv2.imshow("subImg", np.uint8(subImg))
            #cv2.waitKey(0)
            mask[i, j] = mainAngle(subImg, kernelSize, angles, 0.1, sigma=sigma, gamma=gamma, Lambda=Lambda, psi=psi)
    return mask


def makeVecFieldImg(mask, blockLen):
    n, _ = mask.shape
    img = np.full((n * blockLen, n * blockLen), 255, dtype=np.uint8)
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            if mask[i, j] != -1:
                sinAgl = math.sin(mask[i, j] / 180 * math.pi)
                deltaX = sinAgl * blockLen / 2
                P1_x = blockLen // 2 - int(deltaX)
                P2_x = blockLen // 2 + int(deltaX)

                cosAgl = math.cos(mask[i, j] / 180 * math.pi)
                deltaY = cosAgl * blockLen / 2
                P1_y = blockLen // 2 + int(deltaY)
                P2_y = blockLen // 2 - int(deltaY)

                P1 = (j * blockLen + P1_x, i * blockLen + P1_y)
                P2 = (j * blockLen + P2_x, i * blockLen + P2_y)
                cv2.line(img, P1, P2, 0, 3)
    return img



os.chdir("./imgs")
imlist = os.listdir(".")
img_max_number = len(imlist)

# Read image
img = cv2.imread(imlist[0]).astype(np.float32)
img = cv2.resize(img, (600, 600), cv2.INTER_AREA)

# gabor process
out = Gabor_process(img)

angles_count = 15
kernel_size = 15
sigma = 15
gamma = 1.8
Lambda = 10
psi = 0





gabor = Gabor_filter(K_size = kernel_size, Sigma = sigma, Gamma = gamma, Lambda = Lambda, Psi = psi, angle = 0)

gabor = cv2.resize(gabor, (600, 600), cv2.INTER_AREA)
gabor = cv2.normalize(gabor, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
gabor = np.uint8(gabor)
cv2.imshow("gabor", gabor)
mask = turnMask(img, kernel_size, angles_count, sigma=sigma, gamma=gamma, Lambda=Lambda, psi=psi)
retImg = makeVecFieldImg(mask, 20)
cv2.imshow("vec field img", retImg)

#cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
