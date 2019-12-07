import cv2
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    img1Path = sys.argv[1]
    img2Path = sys.argv[2]
    kp1Path = sys.argv[3]
    kp2Path = sys.argv[4]
    matchesPath = sys.argv[5]

    img1 = cv2.imread(img1Path,0)          # queryImage
    img2 = cv2.imread(img2Path,0) # trainImage
    # Initiate ORB detector


    kp1 = []
    kp2 = []
    matches = []
    with open(kp1Path) as fp:
       for line in fp.readlines():
           point = line.strip('\n').split(' ')
           kp1.append(cv2.KeyPoint(x=int(point[1]), y=int(point[0]), _size=31))

    with open(kp2Path) as fp:
       for line in fp.readlines():
           point = line.strip('\n').split(' ')
           kp2.append(cv2.KeyPoint(x=int(point[1]), y=int(point[0]), _size=31))
    i = 0
    with open(matchesPath) as fp:
       for line in fp.readlines():
           point = line.strip('\n')
           matches.append(cv2.DMatch(i, int(point), 0.0))
           i += 1

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
    plt.imshow(img3)
    plt.show()