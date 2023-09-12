# Use nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as the base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set the working directory to /home/appusr
WORKDIR /home/appusr

# Update and install any required packages. For example:
RUN apt-get update && apt-get install -y \
    git \
    vim