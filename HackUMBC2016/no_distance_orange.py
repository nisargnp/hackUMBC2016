#special thanks to http://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

# This is for christian, who's environment is fked
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

# import the necessary packages
import numpy as np
import cv2
 
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    #cv2.imshow("image1", edged)
 
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 0):
        return None
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
IMAGE_PATHS = ["images/macshot.jpg"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
#image = cv2.resize(image, (1280,720))
print image.shape
# image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

#BLUE
# scale_red = 25
# scale_blue = 20
# scale_green = 25
# BINARY_THRESHOLDS = np.array([[200-scale_blue,200-scale_green,150-scale_red], [200+scale_blue,200+scale_green,150+scale_red]])

#ORANGE
scale_red = 75
scale_blue = 30
scale_green = 50
BINARY_THRESHOLDS = np.array([[30-scale_blue,160-scale_green,180-scale_red], [30+scale_blue,160+scale_green,180+scale_red]])

image = cv2.GaussianBlur(image, (5, 5), 0)
image = cv2.inRange(image, *BINARY_THRESHOLDS)

marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
cam = cv2.VideoCapture(0)
while cam.isOpened():
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    ret_val, image = cam.read();
    image = cv2.flip(image, 1)
    # cv2.imshow("image", image)
    # image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

    edited = cv2.GaussianBlur(image, (5, 5), 0)
    edited = cv2.inRange(edited, *BINARY_THRESHOLDS)

    edited_small = cv2.resize(edited, (0,0), fx=0.3, fy=0.3)
    print edited_small.shape
    cv2.imshow("canny", edited_small)

    marker = find_marker(edited)
    # If marker is none, there is no orange thing, dont draw anything
    if (marker is not None):
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
        # draw a bounding box around the image and display it
        box = np.int0(cv2.cv.BoxPoints(marker))
        kreygasm = cv2.imread('images/Kreygasm.png')
        #print box
        #print box[0][0]
        #print box[0][1]
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fft" % (inches / 12),
                    (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        #kreygasm = np.copyto(image.rowRange(min(box[0][0], box[1][0]), max(box[2][0], box[3][0]).colRange(min(box[0][1], box[1][1]), max(box[2][1], box[3][1]))))
        
        #zee = abs(max(box[2][0], box[3][0]) - min(box[0][0], box[1][0]))
        #if (zee <= 0):
        #    zee = 10
        #pee = abs(max(box[2][1], box[3][1]) - min(box[0][1], box[1][1]))
        #if (pee <= 10):
        #    pee = 10

        #kreygasm = cv2.resize(kreygasm, (zee, pee))
        #for i in range(0, kreygasm.shape[0] - 1):
        #    for j in range(0, kreygasm.shape[1] - 1):
        #        gee = kreygasm[i][j]
        #        image[min(box[0][0], box[1][0]) + i][min(box[0][1], box[1][1]) + j] = gee
        #        j += 1
        
        #    i += 1




    # image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
    cv2.namedWindow("image", 0)
    cv2.imshow("image", image)
    print image.shape
    cv2.resizeWindow("image", 1280, 720)
    cv2.waitKey(30)

