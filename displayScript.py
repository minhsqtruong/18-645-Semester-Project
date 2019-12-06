import numpy as np
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('face1.jpg',0)          # queryImage
img2 = cv2.imread('face2.jpg',0) # trainImage
# Initiate ORB detector


kp1 = []
kp2 = []
matches = []
filepath = "out2.txt"
with open(filepath) as fp:
   for line in fp.readlines():
       point = line.strip('\n').split(' ')
       kp1.append(cv2.KeyPoint(x=int(point[1]), y=int(point[0]), _size=31))

# img3 = cv2.drawKeypoints(img1,kp1,None, flags=2)
# plt.imshow(img3)
# plt.show()
filepath = "out2.txt"
with open(filepath) as fp:
   for line in fp.readlines():
       point = line.strip('\n').split(' ')
       kp2.append(cv2.KeyPoint(x=int(point[1]), y=int(point[0]), _size=31))
filepath = "matches.txt"
i = 0
with open(filepath) as fp:
   for line in fp.readlines():
       point = line.strip('\n')
       matches.append(cv2.DMatch(i, int(point), 0.0))
       i += 1

# orb = cv2.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
plt.imshow(img3)
plt.show()