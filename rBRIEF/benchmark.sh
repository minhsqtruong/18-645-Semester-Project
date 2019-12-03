#! /bin/bash
make test
RUN=1000
MAX_WPB=1000
touch tmp.txt
/usr/local/cuda/bin/nvprof --unified-memory-profiling off --log-file tmp.txt ./tb_rBRIEF 1 
time=$(grep -n "gpu_rBRIEF" tmp.txt | cut -d " " -f 3)
rawval=$(echo ${time} | cut -d "m" -f 1)
echo ${rawval}
rm tmp.txt
