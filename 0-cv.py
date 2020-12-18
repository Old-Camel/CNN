import cv2
import numpy as np
from scipy import signal

fn = "test1.jpg"
myimg = cv2.imread(fn)
img = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
srcimg = np.array(img, np.double)
myh = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
myj = signal.convolve2d(srcimg, myh, mode="same")
jgimg = img - myj
# img = cv2.resize(img,None,fx=1.5,fy=1.5)
# jgimg = cv2.resize(jgimg,None,fx=1.5,fy=1.5)

cv2.imshow('src', img)
cv2.imshow('dst', jgimg)
cv2.waitKey()
cv2.destroyAllWindows()