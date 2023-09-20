#!/bin/bash

WORKSPACE=$PWD
CONTAINER_VERSION=23.04
TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
NAME=triton_with_ft_demo

# build docker image
docker build --rm --build-arg TRITON_VERSION=${CONTAINER_VERSION}  -t ${TRITON_DOCKER_IMAGE} -f ./Dockerfile .

# run docker container
docker run -d -it --rm --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=1g \
    -v ${WORKSPACE}:/workspace \
    --name ${NAME} \
    -p 7070:7070 \
    -p 7071:7071 \
    -p 7072:7072 \
    ${TRITON_DOCKER_IMAGE}

docker exec -it ${NAME} /bin/bash
