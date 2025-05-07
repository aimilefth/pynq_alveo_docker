#!/bin/bash
# docker/docker_build.sh

# Build the Docker image, tag it as aimilefth/anomaly_det:latest, and push it
docker build -t aimilefth/pynq_alveo_docker:coroni  --push .