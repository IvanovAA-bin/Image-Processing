import cv2 as cv
import numpy as np
import os

# constants
window_source_name = "source image"
window_green_part_name = "green part image"
window_ill_part_name = "ill part image"
window_trackbar_settings_name = "trackbars settings"
window_watershed_name = "watershed image"
window_filer_settings_name = "filter settings"

low_H_green_name = "low H green"
low_S_green_name = "low S green"
low_V_green_name = "low V green"
high_H_green_name = "high H green"
high_S_green_name = "high S green"
high_V_green_name = "high V green"

low_H_ill_name = "low H ill"
low_S_ill_name = "low S ill"
low_V_ill_name = "low V ill"
high_H_ill_name = "high H ill"
high_S_ill_name = "high S ill"
high_V_ill_name = "high V ill"

image_pick_trackbar_name = "image pick"
image_filter_trackbar_name = "filt type"
image_base_name = "./ListsDataset/"
filters_base_name = "./GeneratedFilters/"
image_pick_name = image_base_name + "1.jpg"
image_pick_number = 1
image_filter_type = 0
current_image_pick_number = -1
current_filter_type = -1

bilateral_diameter_name = "bil diam"
bilateral_diameter_max_value = 50
bilateral_sigmaColor_name = "bil sigColor"
bilateral_sigmaColor_max_value = 200
bilateral_sigmaSpace_name = "bil sigSpace"
bilateral_sigmaSpace_max_value = 200

nlm_h_name = "nlm h"
nlm_h_max_value = 30
nlm_temWinSize_name = "nlm temWinSize"
nlm_temWinSize_max_value = 30
nlm_searWinSize_name = "nlm searWinSize"
nlm_searWinSize_max_value = 50


max_value = 255
max_value_H = 360//2
image_max_number = 2000
# constants

# variables
low_H_green = 40
high_H_green = 100
low_S_green = 22
high_S_green = 244
low_V_green = 33
high_V_green = 250

low_H_ill = 4
high_H_ill = 23
low_S_ill = 56
high_S_ill = 140
low_V_ill = 110
high_V_ill = 201

bilateral_diameter_value = 4
bilateral_sigmaColor_value = 59
bilateral_sigmaSpace_value = 80

nlm_h_value = 4
nlm_temWinSize_value = 9
nlm_searWinSize_value = 20

filtration_acquired = True
# variables


# callback functions for trackbars
def clbk_low_H_green(val):
    global low_H_green
    global high_H_green
    low_H_green = val
    low_H_green = min(high_H_green - 1, low_H_green)
    cv.setTrackbarPos(low_H_green_name, window_trackbar_settings_name, low_H_green)


def clbk_low_S_green(val):
    global low_S_green
    global high_S_green
    low_S_green = val
    low_S_green = min(high_S_green - 1, low_S_green)
    cv.setTrackbarPos(low_S_green_name, window_trackbar_settings_name, low_S_green)


def clbk_low_V_green(val):
    global low_V_green
    global high_V_green
    low_V_green = val
    low_V_green = min(high_V_green - 1, low_V_green)
    cv.setTrackbarPos(low_V_green_name, window_trackbar_settings_name, low_V_green)


def clbk_high_H_green(val):
    global low_H_green
    global high_H_green
    high_H_green = val
    high_H_green = max(high_H_green, low_H_green + 1)
    cv.setTrackbarPos(high_H_green_name, window_trackbar_settings_name, high_H_green)

def clbk_high_S_green(val):
    global low_S_green
    global high_S_green
    high_S_green = val
    high_S_green = max(high_S_green, low_S_green + 1)
    cv.setTrackbarPos(high_S_green_name, window_trackbar_settings_name, high_S_green)

def clbk_high_V_green(val):
    global low_V_green
    global high_V_green
    high_V_green = val
    high_V_green = max(high_V_green, low_V_green + 1)
    cv.setTrackbarPos(high_V_green_name, window_trackbar_settings_name, high_V_green)


def clbk_low_H_ill(val):
    global low_H_ill
    global high_H_ill
    low_H_ill = val
    low_H_ill = min(low_H_ill, high_H_ill - 1)
    cv.setTrackbarPos(low_H_ill_name, window_trackbar_settings_name, low_H_ill)


def clbk_low_S_ill(val):
    global low_S_ill
    global high_S_ill
    low_S_ill = val
    low_S_ill = min(low_S_ill, high_S_ill - 1)
    cv.setTrackbarPos(low_S_ill_name, window_trackbar_settings_name, low_S_ill)


def clbk_low_V_ill(val):
    global low_V_ill
    global high_V_ill
    low_V_ill = val
    low_V_ill = min(low_V_ill, high_V_ill - 1)
    cv.setTrackbarPos(low_V_ill_name, window_trackbar_settings_name, low_V_ill)


def clbk_high_H_ill(val):
    global low_H_ill
    global high_H_ill
    high_H_ill = val
    high_H_ill = max(high_H_ill, low_H_ill + 1)
    cv.setTrackbarPos(high_H_ill_name, window_trackbar_settings_name, high_H_ill)


def clbk_high_S_ill(val):
    global low_S_ill
    global high_S_ill
    high_S_ill = val
    high_S_ill = max(high_S_ill, low_S_ill + 1)
    cv.setTrackbarPos(high_S_ill_name, window_trackbar_settings_name, high_S_ill)


def clbk_high_V_ill(val):
    global low_V_ill
    global high_V_ill
    high_V_ill = val
    high_V_ill = max(high_V_ill, low_V_ill + 1)
    cv.setTrackbarPos(high_V_ill_name, window_trackbar_settings_name, high_V_ill)


def clbk_image_pick(val):
    global image_pick_name
    global image_pick_number
    global lists_paths
    image_pick_number = val
    #image_pick_number = max(1, image_pick_number)
    image_pick_name = image_base_name + lists_paths[image_pick_number]
    #cv.setTrackbarPos(image_pick_trackbar_name, window_trackbar_settings_name, image_pick_number)

def clbk_image_filter_type(val):
    global image_filter_type
    image_filter_type = val


def clbk_bilateral_diameter_value(val):
    global bilateral_diameter_value
    global filtration_acquired
    filtration_acquired = True
    bilateral_diameter_value = val
    bilateral_diameter_value = max(1, bilateral_diameter_value)
    cv.setTrackbarPos(bilateral_diameter_name, window_filer_settings_name, bilateral_diameter_value)


def clbk_bilateral_sigmaColor_value(val):
    global bilateral_sigmaColor_value
    global filtration_acquired
    filtration_acquired = True
    bilateral_sigmaColor_value = val
    bilateral_sigmaColor_value = max(1, bilateral_sigmaColor_value)
    cv.setTrackbarPos(bilateral_sigmaColor_name, window_filer_settings_name, bilateral_sigmaColor_value)


def clbk_bilateral_sigmaSpace_value(val):
    global bilateral_sigmaSpace_value
    global filtration_acquired
    filtration_acquired = True
    bilateral_sigmaSpace_value = val
    bilateral_sigmaSpace_value = max (1, bilateral_sigmaSpace_value)
    cv.setTrackbarPos(bilateral_sigmaSpace_name, window_filer_settings_name, bilateral_sigmaSpace_value)


def clbk_nlm_h_value(val):
    global nlm_h_value
    global filtration_acquired
    filtration_acquired = True
    nlm_h_value = val
    nlm_h_value = max(1, nlm_h_value)
    cv.setTrackbarPos(nlm_h_name, window_filer_settings_name, nlm_h_value)


def clbk_nlm_temWinSize_value(val):
    global nlm_temWinSize_value
    global filtration_acquired
    filtration_acquired = True
    nlm_temWinSize_value = val
    nlm_temWinSize_value = max(1, nlm_temWinSize_value)
    cv.setTrackbarPos(nlm_temWinSize_name, window_filer_settings_name, nlm_temWinSize_value)


def clbk_nlm_searWinSize_value(val):
    global nlm_searWinSize_value
    global filtration_acquired
    filtration_acquired = True
    nlm_searWinSize_value = val
    nlm_searWinSize_value = max(1, nlm_searWinSize_value)
    cv.setTrackbarPos(nlm_searWinSize_name, window_filer_settings_name, nlm_searWinSize_value)
# callback functions for trackbars


lists_paths = os.listdir(image_base_name)
filters_paths = os.listdir(filters_base_name)
image_max_number = len(lists_paths)

# windows settings
cv.namedWindow(window_source_name)
cv.namedWindow(window_green_part_name)
cv.namedWindow(window_ill_part_name)
cv.namedWindow(window_trackbar_settings_name)
cv.namedWindow(window_watershed_name)
cv.namedWindow(window_filer_settings_name)
# windows settings

# trackbars
cv.createTrackbar( low_H_green_name, window_trackbar_settings_name,  low_H_green, max_value_H,  clbk_low_H_green)
cv.createTrackbar(high_H_green_name, window_trackbar_settings_name, high_H_green, max_value_H, clbk_high_H_green)
cv.createTrackbar( low_S_green_name, window_trackbar_settings_name,  low_S_green,  max_value,   clbk_low_S_green)
cv.createTrackbar(high_S_green_name, window_trackbar_settings_name, high_S_green,  max_value,  clbk_high_S_green)
cv.createTrackbar( low_V_green_name, window_trackbar_settings_name,  low_V_green,  max_value,   clbk_low_V_green)
cv.createTrackbar(high_V_green_name, window_trackbar_settings_name, high_V_green,  max_value,  clbk_high_V_green)

cv.createTrackbar( low_H_ill_name, window_trackbar_settings_name,  low_H_ill, max_value_H,  clbk_low_H_ill)
cv.createTrackbar(high_H_ill_name, window_trackbar_settings_name, high_H_ill, max_value_H, clbk_high_H_ill)
cv.createTrackbar( low_S_ill_name, window_trackbar_settings_name,  low_S_ill,  max_value,   clbk_low_S_ill)
cv.createTrackbar(high_S_ill_name, window_trackbar_settings_name, high_S_ill,  max_value,  clbk_high_S_ill)
cv.createTrackbar( low_V_ill_name, window_trackbar_settings_name,  low_V_ill,  max_value,   clbk_low_V_ill)
cv.createTrackbar(high_V_ill_name, window_trackbar_settings_name, high_V_ill,  max_value,  clbk_high_V_ill)

cv.createTrackbar(image_pick_trackbar_name, window_trackbar_settings_name, image_pick_number, image_max_number, clbk_image_pick)
cv.createTrackbar(image_filter_trackbar_name, window_trackbar_settings_name, image_filter_type, 2,
                  clbk_image_filter_type)

cv.createTrackbar(bilateral_diameter_name, window_filer_settings_name, bilateral_diameter_value,
                  bilateral_diameter_max_value, clbk_bilateral_diameter_value)
cv.createTrackbar(bilateral_sigmaColor_name, window_filer_settings_name, bilateral_sigmaColor_value,
                  bilateral_sigmaColor_max_value, clbk_bilateral_sigmaColor_value)
cv.createTrackbar(bilateral_sigmaSpace_name, window_filer_settings_name, bilateral_sigmaSpace_value,
                  bilateral_sigmaSpace_max_value, clbk_bilateral_sigmaSpace_value)
cv.createTrackbar(nlm_h_name, window_filer_settings_name, nlm_h_value, nlm_h_max_value, clbk_nlm_h_value)
cv.createTrackbar(nlm_temWinSize_name, window_filer_settings_name, nlm_temWinSize_value, nlm_temWinSize_max_value,
                  clbk_nlm_temWinSize_value)
cv.createTrackbar(nlm_searWinSize_name, window_filer_settings_name, nlm_searWinSize_value, nlm_searWinSize_max_value,
                  clbk_nlm_searWinSize_value)
# trackbars

markerType = True

#cv.resizeWindow(window_trackbar_settings_name, 400, 200)
source_image = None
while True:
    if current_image_pick_number != image_pick_number:
        source_image = cv.imread(image_pick_name)
        current_image_pick_number = image_pick_number
        current_filter_type = -1

    if source_image is None:
        break

    if filtration_acquired:
        current_filter_type = -1

    if current_filter_type != image_filter_type:
        current_filter_type = image_filter_type
        filtration_acquired = False
        source_image = cv.imread(image_pick_name)
        if image_filter_type == 1:
            source_image = cv.bilateralFilter(source_image, bilateral_diameter_value, bilateral_sigmaColor_value,
                                              bilateral_sigmaSpace_value)
        elif image_filter_type == 2:
            source_image = cv.fastNlMeansDenoisingColored(source_image, None, nlm_h_value, nlm_h_value,
                                                          nlm_temWinSize_value, nlm_searWinSize_value)

    image_HSV = cv.cvtColor(source_image, cv.COLOR_BGR2HSV)
    image_green_part = cv.inRange(image_HSV, (low_H_green, low_S_green, low_V_green),
                                             (high_H_green, high_S_green, high_V_green))
    image_ill_part = cv.inRange(image_HSV, (low_H_ill, low_S_ill, low_V_ill),
                                           (high_H_ill, high_S_ill, high_V_ill))

    markers = np.zeros((source_image.shape[0], source_image.shape[1]), dtype="int32")

    if markerType:
        morph_element = cv.getStructuringElement(cv.MORPH_RECT, (17, 17))
        dilation_added_image = cv.dilate(cv.add(image_ill_part, image_green_part), morph_element)
        markers[dilation_added_image == 0] = 255
    else:
        markers[0:10, 0:10] = 255
        markers[0:10, markers.shape[1] - 10:markers.shape[1]] = 255
        markers[markers.shape[0] - 10:markers.shape[0], 0:10] = 255
        markers[markers.shape[0] - 10:markers.shape[0], markers.shape[1] - 10:markers.shape[1]] = 255
    markers[image_green_part > 0] = 60
    markers[image_ill_part > 0] = 30

    testing = cv.watershed(source_image, markers)
    mask = np.zeros_like(source_image, np.uint8)
    mask[testing == 30] = (0, 0, 255)
    mask[testing == 60] = (255, 0, 255)

    cv.imshow(window_source_name, source_image)
    cv.imshow(window_green_part_name, image_green_part)
    cv.imshow(window_ill_part_name, image_ill_part)
    cv.namedWindow(window_trackbar_settings_name)
    cv.imshow(window_watershed_name, mask)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
    if key == 97:
        image_pick_number -= 1
        image_pick_number = max(0, image_pick_number)
        cv.setTrackbarPos(image_pick_trackbar_name, window_trackbar_settings_name, image_pick_number)
    if key == 100:
        image_pick_number += 1
        image_pick_number = min(image_pick_number, image_max_number)
        cv.setTrackbarPos(image_pick_trackbar_name, window_trackbar_settings_name, image_pick_number)
    if key == ord('x'):
        markerType = not markerType