#!/bin/bash

ROOT=`dirname $0`
TAG="toyota_collision_detection"

docker build --force-rm -f $ROOT/Dockerfile -t $TAG $ROOT

