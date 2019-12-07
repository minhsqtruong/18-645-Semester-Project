# oFAST
This is the oFAST keypoint detector part of the ORB pipeline. `tb_oFAST.cu` reads in an image from `image.h`. It can either output the coordinates of all the keypoints and/or the patch of all the keypoints.

### Make instruction
* In `tb_oFAST.cu`, go to line 44 and comment out the `printf` statement to print all the patches and/or go to line 50 and comment out `printf` statement to print all the coordinates
* In this folder, run `make test`
* run `./tb_oFAST`
