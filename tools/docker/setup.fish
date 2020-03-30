#!/bin/fish

set DOCKER_IMAGE defeatcovid:0.1

if test (docker images -q $DOCKER_IMAGE 2> /dev/null) = "";
   echo "Building docker image..."
   docker build --tag defeatcovid:0.1 .
else
    echo "Docker image already exists! Skipping build phase..."
end

function is_repo_root
    if not test -e Dockerfile;
        echo "This needs to be source from the root of the repository!"
        exit 2
    end
end

is_repo_root

set WD $PWD

function dkrun
    if not test -n "$argv";
        echo "Script to run missing!"
        return
    end
    if not is_repo_root;
        return
    end
    echo "docker run -ti --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v $WD:$WD -w $WD $DOCKER_IMAGE python3 $WD/$argv"
    docker run -ti --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v $WD:$WD -w $WD $DOCKER_IMAGE python3 $WD/$argv
end