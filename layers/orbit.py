"""
Author by Sri Harsha Konuru 

This module has the functions for segmentation and diameter calculations
Visualizations are added on the output image in the pipeline

"""

import cv2
import numpy as np

from layers import config

def segment_object(img):

    """ 
    Generic opencv version of segmentation of cookie from background
    - Works with multiple cookies as well


    Pros: Light weight, Fast computation 
    Cons: Prone to noise, shadows

    Parameters
    ----------------
    img : input RGB image to be passed RGB 'not the cv2 input BGR version'

    Returns
    ----------------
    final_contours: List of contours of the cookies 
    (w, h) : Horizontal and vertical thickness of cookie respectively

    """

    #Preprocessing of image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),3)

    #Otsu thresholding for dynamic boundary creation 
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
    
    
    #Morphological operations to remove speckle noise after threshold
    kernel = np.ones((5,5),np.uint8)
    new_edges = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contour, _ = cv2.findContours(new_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []

    for e, cnt in enumerate(contour):

        area = cv2.contourArea(contour[e])
        x,y,w,h = cv2.boundingRect(contour[e])
        rect_area = w*h
        extent = float(area)/rect_area

        """
        All the noises are not completely eliminated, Using different approches for different types of noises 
        Condition 1: 
            -- cv2.contourArea(contour[e])> 5000
                    This conditions takes care of small contours which are cookie particles spread as noise 

        Condition 2:
            -- area / rect_area extent > 0.50
                    This condition takes care of contours which are not round 

        Condition 3:
            -- (area/(img.shape[0] * img.shape[1]))< 0.7
                    This condition takes care of very large contours formed because of the changes in the background 
                    This condition breaks when cookie covers 70% of the image portion - Restiction on operations perspective 
                    DO NOT PASS CROPPED IMAGES

        Condition 4: 
            -- min(w,h)/max(w,h) > 0.5
                    Mostly cookie is round, so l and w should be almost equal, when one is 50% of other something is terribly wrong
                    I used 50% so that broken cookie should not be rejected
        """
        if cv2.contourArea(contour[e])> 5000 and extent > 0.50 and (area/(img.shape[0] * img.shape[1]))< 0.7 and min(w,h)/max(w,h) > 0.5:
            final_contours.append(contour[e])
        
    final_contours = np.asarray(final_contours)

    return final_contours, (w * config.cm_per_pixel , h * config.cm_per_pixel)