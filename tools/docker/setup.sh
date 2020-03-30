#!/bin/bash

DOCKER_IMAGE=defeatcovid:0.1

if [[ "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
   docker build --tag defeatcovid:0.1 .
fi

is_repo_root() {
    if [ ! -f "Dockerfile" ]; then
        echo "This needs to be source from the root of the repository!"
        return
    fi
}

is_repo_root

WD=$PWD

dkrun() {
    if [[ -z "$1" ]]; then
        echo "Script to run missing!"
        return
    fi
    if [ ! is_repo_root ]; then
        return
    fi
    docker run -it -v $PWD:$PWD $DOCKER_IMAGE python3 $PWD/$1
    echo "docker run -ti --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v $WD:$WD -w $WD $DOCKER_IMAGE python3 $WD/$1"
    docker run -ti --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v $WD:$WD -w $WD $DOCKER_IMAGE python3 $WD/$1
}