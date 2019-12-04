import os
RUN = 1000
sum = 0.0
WPB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
for wpb in WPB:
    sum = 0.0
    for run in range(1,RUN):
        os.system("/usr/local/cuda/bin/nvprof --log-file tmp.txt --unified-memory-profiling off ./tb_rBRIEF " + str(wpb))
        with open('./tmp.txt') as log:
            for line in log:
                if ("gpu_rBRIEF_naive" in line):
                    val = float(line.split()[1].split("m")[0])
                    sum += val
    print("Average for " + str(wpb) + " is " + str(sum / RUN))

