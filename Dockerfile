# Use an official Python runtime as a parent image
FROM python:3.6.5
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3-pip python3-dev && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
