import os
RUN = 50
sum = 0.0
for run in range(1,RUN):
    os.system("/usr/local/cuda/bin/nvprof --log-file tmp.txt --unified-memory-profiling off ./a.out ")
    with open('./tmp.txt') as log:
        for line in log:
            if ("calcKeyPoints" in line):
                val = float(line.split()[1][:-2])
                sum += val
print("Average is " + str(sum / RUN))