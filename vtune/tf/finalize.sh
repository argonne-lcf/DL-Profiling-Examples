#!/bin/sh
#COBALT -t 60
#COBALT -n 8
#COBALT -q debug-flat-quad
#COBALT -A datascience



#module load /soft/datascience/conda/miniconda3/4.6.14/modulefile
#module load gcc/7.3.0
module unload darshan
module load datascience/pytorch-1.1
module swap intel/18.0.0.128 intel/19.0.3.199

BASE=/projects/datascience/parton/atlasml/atlas-pointnet
echo [$SECONDS] BASE=$BASE
WORKDIR=/projects/datascience/parton/atlasml/test_atlas-pointnet
echo [$SECONDS] WORKDIR=$WORKDIR
RANKS_PER_NODE=1
#INPUT="-i /projects/datascience/parton/atlasml/test_atlas-pointnet/353778/model_00002_00170.torch_model_state_dict"
#CPROFILE="-m cProfile -o $WORKDIR/${COBALT_JOBID}.cprofile"



#export AMPLXE_RUNTOOL_OPTIONS=--no-altstack
export PE_RANK=$ALPS_APP_PE
export PMI_NO_FORK=1
export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=1
#export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
#export MKL_DYNAMIC=FALSE

# Executing this inside of an interactive session is as easy as calling your python script with singularity:
#IMG="singularity exec -B /projects/datascience/parton -B /projects/atlasMLbjets/parton /soft/datascience/singularity/conda_images/PyTorch_SparseConvNet.simg"
#MPROF="mprof run -o ${COBALT_JOBID}.mprof"
#VTUNE="amplxe-cl -collect advanced-hotspots -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  --"
#VTUNE="amplxe-cl -collect concurrency -knob analyze-openmp=true -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  --"
amplxe-cl -finalize \
   -search-dir /lus/theta-fs0/software/datascience/pytorch/1.1.0/lib/python3.5/site-packages/torch/lib/ \
   -search-dir /opt/cray/udreg/2.3.2-6.0.7.1_5.13__g5196236.ari/lib64 \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib/python3.5/site-packages/numpy/core/ \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib/python3.5/site-packages/numpy/random/ \
   -search-dir /sbin/ \
   -search-dir /lib/ \
   -search-dir /lib64/ \
   -search-dir /usr/lib64 \
   -search-dir /usr/lib \
   -search-dir /bin \
   -search-dir /opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/ \
   -search-dir /opt/gcc/7.3.0/snos/lib64/ \
   -search-dir lus/theta-fs0/software/buildtools/trackdeps/lib64/ \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib/python3.5/site-packages/numpy/core \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib \
   -search-dir /opt/cray/pe/mpt/7.7.3/gni/mpich-gnu/7.1/lib \
   -search-dir /lus/theta-fs0/software/buildtools/trackdeps/lib64 \
   -search-dir /lus/theta-fs0/software/datascience/horovod/0.16.3-nomlsl/lib/python3.5/site-packages/horovod-0.16.3-py3.5-linux-x86_64.egg/horovod/torch \
   -search-dir /opt/intel/python/2017.0.035/intelpython35/lib/python3.5/site-packages/pandas/ \
   -search-dir /gpfs/mira-home/parton/.local/lib/python3.5/site-packages/google/protobuf/pyext/ \
   -search-dir /opt/cray/ugni/6.0.14.0-6.0.7.1_3.13__gea11d3d.ari/lib64 \
   -r $1  
