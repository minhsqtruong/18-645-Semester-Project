all: benchmark

benchmark: gpu_benchmark.cu
	nvcc gpu_benchmark.cu -o benchmark

clean:
	rm -rf benchmark
