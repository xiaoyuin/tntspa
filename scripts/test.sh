
DDIR=../data/monument_600
MDIR=../output/models

if [ -n "$1" ]
    then DDIR=$1
fi

if [ -n "$2" ]
    then MDIR=$2
fi

echo $DDIR
echo $MDIR