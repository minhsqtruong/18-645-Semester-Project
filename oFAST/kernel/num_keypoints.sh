for i in {10..300..10}
do nvprof --unified-memory-profiling off ./a.out $i
done 