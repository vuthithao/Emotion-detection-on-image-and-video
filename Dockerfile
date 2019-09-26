FROM python:3.5

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY . /

WORKDIR /
RUN pip3 install -r requirement.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python3", "server.py"]
