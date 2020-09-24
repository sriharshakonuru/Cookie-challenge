"""
Author by Sri Harsha Konuru 

This module has the functions for segmentation of chocolate chips
The area threshold of chocolate chip defines the number of cookies detected

"""

import cv2
import numpy as np

from layers import config

def detect_chocolate(image, cookieArea):

    """ 
    Generic opencv version of segmentation of chocolate chips in the cookie
    - Works with cookie at constant height and constant light source


    Pros: Light weight, Fast computation 
    Cons: Prone to noise, shadows

    Parameters
    ----------------
    img : input RGB image to be passed RGB 'not the cv2 input BGR version'
    area : Area of the whole cookie

    Returns
    ----------------
    Image: Overlayed segmented portions of the chocolate chips
    n_chips: Number of chips in the cookie - defined by percentage of the entire cookie 
            (Using 5% hard threshold for my data)

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

    # Bilateral Filter to reduce the cracks on the chip
    image = cv2.bilateralFilter(image,30,150, 150)
    x,y,z = cv2.split(image)

    # Contrast enhancement on XYZ color space - pretty powerful contrast 
    contrast_enh = x*y/z
    contrast_enh = (contrast_enh/np.nanmax(contrast_enh)) * 255.
    contrast_enh = np.uint8(contrast_enh)  # Normalization
    
    _,th1 = cv2.threshold(contrast_enh,100,255,cv2.THRESH_BINARY)
    canny = cv2.Canny(th1,120,200)

    # Morphological closing operation removes most of the small noise 
    kernel = np.ones((7, 7),np.uint8)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    contour,_ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    final_contours = []
    for cnt in contour:

        # Considering the contours which are more than 5% area of the entire cookie 
        if (cv2.contourArea(cnt)/cookieArea) * 100 > config.chocolate_area_threshold:
            final_contours.append(cnt)

    output = image.copy()
    cv2.drawContours(output, final_contours, -1, (255,0,0), 5)

    return output, len(final_contours)




