#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  slim    Run the slim version of the Docker container."
    echo "  gpu     Run the GPU-enabled version of the Docker container."
    echo "  help    Display this help message."
    exit 1
}

# Check if help is requested
if [[ "$1" == "help" ]]; then
    usage
fi

# Initialize variables with default values
IMAGE_TAG="aimilefth/pynq_alveo_docker:coroni"
# Current working directory
CURRENT_PATH=$(pwd)
# Docker run mode
DETACHED="-it"


# Function to collect Docker device options
collect_docker_devices() {
    local devices=""

    echo "Collecting Docker device options..." >&2

    # Find xclmgmt devices
    local xclmgmt_drivers
    xclmgmt_drivers=$(find /dev -name "xclmgmt*" 2>/dev/null)
    for device in ${xclmgmt_drivers}; do
        devices+="--device=${device} "
    done

    # Find renderD devices
    local render_drivers
    render_drivers=$(find /dev/dri -name "renderD*" 2>/dev/null)
    for device in ${render_drivers}; do
        devices+="--device=${device} "
    done

    echo "Docker devices collected: ${devices}" >&2
    echo "${devices}"
}

# Collect Docker device options
docker_devices=$(collect_docker_devices)

# Define Docker run parameters with conditional GPU option
docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v /etc/xbutler:/etc/xbutler \
    -v /scrape:/scrape \
    --privileged \
    -v ${CURRENT_PATH}/src:/app/src \
    -v ${CURRENT_PATH}/mounted_dir:/app/mounted_dir \
    -v ${CURRENT_PATH}/outputs:/app/outputs \
    -p 8080:8080 \
    --pull=always \
    --network=host \
    ${DETACHED} \
    ${IMAGE_TAG}
END
)

echo "Docker run parameters configured."

# Execute the Docker run command
echo "Running Docker container with image: ${IMAGE_TAG}"
docker run \
  ${docker_devices} \
  ${docker_run_params}