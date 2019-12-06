import os
for run in range(10,30,10):
    os.system("/usr/local/cuda/bin/nvprof --log-file tmp.txt --unified-memory-profiling off ./a.out %d" %(run))
    with open('./threshold.txt') as log:
        for line in log:
            if ("calcKeyPoints" in line):
                val = float(line.split()[1][:-2])
                print(run, val)