#special thanks to http://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

# This is for christian, who's environment is fked
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

# import the necessary packages
import numpy as np
import cv2

def fill_contour(image):

    # Copy the thresholded image.
    im_floodfill = image.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv

    return im_out
 
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cv2.imshow("image1", edged)

 
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 12.0
 
# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
KNOWN_WIDTH = 2.0
 
# initialize the list of images that we'll be using
IMAGE_PATHS = ["images/box.jpg"]
# IMAGE_PATHS = ["images/g1.jpg", "images/g2.jpg", "images/g3.jpg"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
# image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

# BLUE
#scale_red = 25
#scale_blue = 20
#scale_green = 25
#BINARY_THRESHOLDS = np.array([[200-scale_blue,200-scale_green,150-scale_red], [200+scale_blue,200+scale_green,150+scale_red]])

# ORANGE
# scale_red = 50
# scale_blue = 5
# scale_green = 25
# BINARY_THRESHOLDS = np.array([[5-scale_blue,120-scale_green,220-scale_red], [5+scale_blue,120+scale_green,220+scale_red]])

# GREEN
# scale_red = 5
# scale_blue = 5
# scale_green = 5
# BINARY_THRESHOLDS = np.array([[65-scale_blue,250-scale_green,5-scale_red], [65+scale_blue,250+scale_green,5+scale_red]])

image = cv2.GaussianBlur(image, (5, 5), 0)
#image = cv2.inRange(image, *BINARY_THRESHOLDS)


marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:

    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera

    image = cv2.imread(imagePath)
    # image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

    edited = cv2.GaussianBlur(image, (5, 5), 0)

   # edited = cv2.inRange(edited, *BINARY_THRESHOLDS)

    edited_small = cv2.resize(edited, (0,0), fx=0.3, fy=0.3)
    cv2.imshow("canny", edited_small)
    #cv2.waitKey(0)

    marker = find_marker(edited)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
    # draw a bounding box around the image and display it
    box = np.int0(cv2.cv.BoxPoints(marker))
    cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
    cv2.putText(image, "%.2fft" % (inches / 12),
                (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)
    
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
    cv2.namedWindow("image", 0)
    cv2.imshow("image", image)
    cv2.resizeWindow("image", 1280, 720)
    cv2.waitKey(0)