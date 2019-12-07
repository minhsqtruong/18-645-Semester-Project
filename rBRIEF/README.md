# rBRIEF

This is the rBRIEF + matching portion of the ORB pipeline.

### Make instruction
* In `rBRIEF.cuh`, go to line 8 and uncomment out the `#define` statement to print all debug print statemets
* In this folder, run `make test`
* run `./tb_rBRIEF <images per block (1 to 1000)>` 

### Verification
To verify the implemnetaion correctness, enable all debug print as mentioned above, then run the executable. The final output
should be a string of roughly increasing number from 0 to 128. This is the print out of the mapping from 1 image to itself. The
ideal result should be exactly an increasing string from 0 to 128. However, there can be multiple values with the same minimum. Data race
will occur in such case and the last thread that write to the map is the one that is displayed.

### Benchmarking
To benchmark, do `make test` then `python3 benchmark.py`

### Visualization
Run python script to display matches between two pictures with the command

`python3 displayScript.py face1.jpg face2.jpg kp1.txt kp2.txt out.txt`

The first two arguments are the images \
The second two arguments are txt files containing the keypoint coordinates for the respective images \
The last argument is the file containing the matches, the output of the rBrief + match kernel  \
Script is dependent on opencv 
