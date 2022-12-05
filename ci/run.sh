#!/bin/sh

show_usage()
{
    echo "BlackBox Test Driver v1.0"
    echo "Usage: [[-cores=#n] [-a|-b|-bb] [-help]]"
}

SCRIPT_DIR=$(dirname "$0")
ROOT_HOME=$SCRIPT_DIR/..
JOBS= `cat /proc/cpuinfo| grep "processor"| wc -l`/2 | bc
CORES=1
BUILD_CSIM=0
BUILD_BENCH=0

for i in "$@"
do
case $i in
    -cores=*)
        CORES=${i#*=}
        shift
        ;;
    -a)
        BUILD_CSIM=1
        BUILD_BENCH=1
        shift
        ;;
    -b)
        BUILD_CSIM=1
        shift
        ;;
    -bb)
        BUILD_BENCH=1
        shift
        ;;
    -help)
        show_usage
        exit 0
        ;;
    *)
    show_usage   
    exit -1       
    ;;
esac
done

if [ $BUILD_CSIM -eq 1 ];then
    CONFIGS="-DNUM_CORES=$CORES $CONFIGS"
    echo "initialization..."
    echo "make -C $ROOT_HOME clean"
    make -C $ROOT_HOME clean

    echo "build cmodel: CONFIGS="$CONFIGS" make -C $ROOT_HOME"
    CONFIGS=$CONFIGS make -C $ROOT_HOME -j $JOBS
fi

if [ $BUILD_BENCH -eq 1 ];then
    make -C $ROOT_HOME/benchmarks -j $JOBS
fi

echo "running benchmark..."
python regression_test.py
status=$?

exit $status