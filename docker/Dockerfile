FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG XRT_URL=https://www.xilinx.com/bin/public/openDownload?filename=xrt_202210.2.13.466_20.04-amd64-xrt.deb
ARG XRT_NAME=xrt_202210.2.13.466_20.04-amd64-xrt.deb

# APT Installs
RUN apt update && \
    apt install curl git python3-pip -y

# Set working directory inside the container
WORKDIR /app

# Install XRT
RUN curl -L -o ${XRT_NAME} ${XRT_URL} && \
    apt-get update && \
    apt install ./${XRT_NAME} -y && \
    rm ${XRT_NAME}

# Setup
ARG SETUP_DIR_ARG=/setup_dir
ARG SETUP_FILE_ARG=setup.sh
ENV SETUP_DIR=${SETUP_DIR_ARG}
ENV SETUP_FILE=${SETUP_FILE_ARG}

# Create setup directory and copy setup file
RUN mkdir -p ${SETUP_DIR_ARG}
COPY ${SETUP_FILE} ${SETUP_DIR_ARG}

# Modify permissions and ownership of setup file and working directory
RUN chmod u+x ${SETUP_DIR_ARG}/${SETUP_FILE_ARG}

# Install python libraries
RUN python3 -m pip install --upgrade pip
# Copy requirements.txt to the container
COPY ../requirements.txt /tmp/requirements.txt
# Use uv to install Python dependencies
RUN python3 -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
# Marimo config. 
COPY .marimo.toml /root/

# Set PYTHONPATH to ensure /app is on sys.path
ENV PYTHONPATH=/app

EXPOSE 8080

CMD bash -c "source ${SETUP_DIR}/${SETUP_FILE} && marimo edit --host '0.0.0.0' --port '8080' /app/src/marimo/marimo.py"
