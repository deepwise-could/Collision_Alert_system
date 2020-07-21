#!/bin/bash

if `which nvidia-docker > /dev/null`; then 
  DOCKER=nvidia-docker
 else
  DOCKER=docker
fi

CONTAINER=toyota_collision_detection
DOCKER_FLAG=0
DOCKER_ARGS=-p8888:8888
COMMAND=/bin/bash

CUR_ROOT=`pwd`
ROOT=`dirname $0`/..

cd $ROOT
ROOT=`pwd`
cd $CUR_ROOT

for arg in "$@"; do
  if   [ "$arg" == "--docker" ];  then DOCKER_FLAG=1
  elif [ "$DOCKER_FLAG" == "1" ]; then
    DOCKER_ARGS="$DOCKER_ARGS "$arg
    DOCKER_FLAG=0
  else
    COMMAND="$COMMAND '$arg'"
  fi

done

eval $DOCKER run     \
              -it                \
              -v$ROOT:/workspace        \
              -w/workspace              \
              $DOCKER_ARGS       \
              $CONTAINER         \
              $COMMAND
