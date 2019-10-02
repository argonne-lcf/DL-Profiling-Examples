#!/bin/sh
#COBALT -t 60
#COBALT -n 2
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname vtune-ml-workshop


# first argument passed should be the python script you want to run
app=$1

# darshan is loaded at log-in time
# it is a low-level monitoring tool so ALCF can meansure application statistics and adapt to users apps
# darshan does not play well with VTUNE, so we remove it
module unload darshan
# VTUNE binaries are loaded via this command
module swap intel/18.0.0.128 intel/19.0.3.199
# all the python environment comes from here
module load miniconda-3
# list all the loaded modules here for logging history
module list

# where our app is running
echo [$SECONDS] PWD = $PWD

# total number of nodes we have
NUM_NODES=$COBALT_PARTSIZE
# how many MPI ranks per node to run
RANKS_PER_NODE=1
# total MPI ranks to run
TOTAL_RANKS=$(($NUM_NODES * $RANKS_PER_NODE))

# Tensorflow & Torch both use Intel's MKL library for 
# accelerating math operations. MKL uses OpenMP for threading
# specifying the number of threads for each MPI Rank to use
export OMP_NUM_THREADS=64 # $(( 32 / $RANKS_PER_NODE ))
echo [$SECONDS] OMP_NUM_THREADS = $OMP_NUM_THREADS

# special flag that improves MKL performance on Intel KNL
export KMP_BLOCKTIME=0

# special flags to use when running VTUNE (comment out otherwise)
export PE_RANK=$ALPS_APP_PE
export PMI_NO_FORK=1
export OMP_AFFINITY=compact,granularity=core
export OMP_STACKSIZE=16G
export AMPLXE_RUNTOOL_OPTIONS=--no-altstack
ulimit -s unlimited
export DARSHAN_DISABLE=1
LD_PRELOAD=

# extra verbose output from MKL and MKL-DNN when needed
#export MKLDNN_VERBOSE=1
#export MKL_VERBOSE=1

# log the current environment
env > ${COBALT_JOBID}.env


#VTUNE="amplxe-cl -collect advanced-hotspots -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  --"
VTUNE="amplxe-cl -collect hotspots  -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  --"
#VTUNE="amplxe-cl -collect hotspots -knob sampling-mode=hw -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0 -resume-after=300 -duration=500  --"
#VTUNE="amplxe-cl -collect hpc-performance -knob analyze-openmp=true -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  --"
#VTUNE="amplxe-cl -collect memory-consumption -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0   --"
echo [$SECONDS] running VTUNE command setup: $VTUNE

echo [$SECONDS] running application $app

# execute command on Theta
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE --cc none $VTUNE $(which python) $app

echo [$SECONDS] application exited with return code $?

echo [$SECONDS] if successful run, you can now run "./finalize.sh ${COBALT_JOBID}_amplxe" to finalize the analysis directory


