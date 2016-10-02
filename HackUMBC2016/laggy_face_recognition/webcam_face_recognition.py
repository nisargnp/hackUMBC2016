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

def find_face(image):
    # Get user supplied values
    #imagePath = sys.argv[1]
    #cascPath = sys.argv[2]

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the image
    #image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.namedWindow("image", 0)
    cv2.imshow("image", image)
    print image.shape
    cv2.resizeWindow("image", 1280, 720)
    cv2.waitKey(30)

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

# image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

#BLUE
# scale_red = 25
# scale_blue = 20
# scale_green = 25
# BINARY_THRESHOLDS = np.array([[200-scale_blue,200-scale_green,150-scale_red], [200+scale_blue,200+scale_green,150+scale_red]])


# loop over the images
cam = cv2.VideoCapture(0)
while cam.isOpened():
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    ret_val, image = cam.read();
    
    find_face(image)


