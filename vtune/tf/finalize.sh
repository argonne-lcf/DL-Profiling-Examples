#!/bin/sh

# first argument passed should be the directory created by VTUNE
datadir=$1

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



echo [$SECONDS] running VTUNE
amplxe-cl -finalize -r $datadir > ${datadir}.log1 2>&1

ARGS=$(./finalize.py ${datadir}.log1)
echo [$SECONDS] ARGS= $ARGS
amplxe-cl -finalize -r $datadir $ARGS

echo [$SECONDS] application exited with return code $?

echo [$SECONDS] if successful run, you can now run "amplxe-gui $datadir" to view the report, if you cannot get a window to launch, try installing VTUNE on your local machine and copying the files to your local machine to run there.


