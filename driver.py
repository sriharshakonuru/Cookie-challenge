"""
Author by Sri Harsha Konuru 

Main module runing the entire module 

"""

import cv2
import numpy as np
import glob

from layers import orbit, chocolate, defect

##############################################################
##  THIS CODE RUNS DEMO ON COOKIE IMAGE ######################

data = glob.glob('./Data/raw_photos/*.jpg')

idx = 13
image = cv2.imread(data[idx])
print(" Image is loaded ")

im_crops = []
final_contours, (w, h) = orbit.segment_object(image.copy())
for e, cnt in enumerate(final_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    roi = image[y:y+h, x:x+w]
    im_crops.append(roi)

print(" Segmentation of image: COMPLETED")
print("Horizontal and vertical diameter of cookie is {} and {}".format(w,h))

for e, img in enumerate(im_crops):
    img = img[:,:,::-1]
    output , chips = chocolate.detect_chocolate(img.copy(), cv2.contourArea(final_contours[e]))

    print("Number of chocolate chips present in {} cookie are {}".format(e, chips))

    hexcodes = defect.colorPicker(img.copy())
    print("Top hexcodes in the {} cookie present are {}".format(e, hexcodes))

print("End of CODE ")

