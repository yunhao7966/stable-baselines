#!/bin/bash

CPU_PARENT=ubuntu:16.04
GPU_PARENT=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

TAG=stablebaselines/stable-baselines
VERSION=v3.0.0

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  TENSORFLOW_PACKAGE="tensorflow-gpu"
else
  PARENT=${CPU_PARENT}
  TENSORFLOW_PACKAGE="tensorflow"
  TAG="${TAG}-cpu"
fi

docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg TENSORFLOW_PACKAGE=${TENSORFLOW_PACKAGE}\
 -t ${TAG}:${VERSION} .
