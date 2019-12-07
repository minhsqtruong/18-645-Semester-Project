# rBRIEF
This is the rBRIEF + matching portion of the ORB pipeline.

Run python script to display matches between two pictures with the command

`python3 displayScript.py face1.jpg face2.jpg kp1.txt kp2.txt out.txt`


The first two arguments are the images \
The second two arguments are txt files containing the keypoint coordinates for the respective images \
The last argument is the file containing the matches, the output of the rBrief + match kernel  \
Script is dependent on opencv 
