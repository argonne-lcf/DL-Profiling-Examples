#!/bin/bash
#COBALT -n 4
#COBALT -t 1:00:00
#COBALT -q training --attrs mcdram=cache:numa=quad
#COBALT -A SDL_Workshop -O tensorflow_imagenet

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
module load datascience/tensorflow-1.14

PROC_PER_NODE=1

#note: Setting threads to a low count for demonstration purposes
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE \
    -j 2 -d 128 -cc depth \
    -e OMP_NUM_THREADS=8 \
    -e KMP_BLOCKTIME=0 \
    -e HOROVOD_TIMELINE=./timeline.json \
    -e TIMELINE_MARK_CYCLES=1 \
    python tensorflow_synthetic_benchmark.py --num_intra=8 --num_inter=2 --num-iters=10 --device cpu

