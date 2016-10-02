import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

MY_PATH = '../better_cars/'

files = [f for f in listdir(MY_PATH) if isfile(join(MY_PATH, f))]

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# frame = cv2.imread('maxresdefault.jpg')
# print frame
# print frame.shape
# cv2.imshow('image', frame)
# cv2.waitKey(1)


for filename in files:

    print filename

    frame = cv2.imread(MY_PATH + filename)

    cv2.imshow('image', frame)

    if cv2.waitKey(1000) & 0xf == 27:
        break

cv2.destroyAllWindows()

print "Hello World"