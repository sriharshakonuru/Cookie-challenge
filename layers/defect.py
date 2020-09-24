"""
Author by Sri Harsha Konuru 

This module has the functions for burnt and broken cookies
Clustering algorithm for returning hex codes for burnt cookie

"""

from sklearn.cluster import KMeans
import binascii
import numpy as np

from layers import config

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster

	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one

	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist


def colorPicker(image):
    
    """ 

    Source - https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/


    Generic opencv version of return top hex codes of image

    Pros: Light weight, Fast computation 
    Cons: Prone to noise, shadows

    Parameters
    ----------------
    img : input RGB image to be passed RGB 'not the cv2 input BGR version'

    Returns
    ----------------
    hex_codes: Top 3 high frequent hexcodes of pixels

    """

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Clustering of pixels on the image using centroid 
    clt = KMeans(n_clusters = config.n_clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
    

    # Calculating the frequency of each pixel and sorted by the frequency
    startX = 0
    frequency = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
       
        hexcode = binascii.hexlify(bytearray(int(c) for c in color)).decode('ascii')
        endX = startX + (percent * 100)
        frequency[hexcode] = endX - startX
        startX = endX

    listofTuples = sorted(frequency.items() , reverse=True,  key=lambda x: x[1])

    # Gathering hex codes of high frequency pixels - Leave out the first (background)
    hex_codes = []
    for elem in range(1,4):
        hex_codes.append(listofTuples[elem][0])

    return hex_codes
   

