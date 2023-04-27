FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

#RUN sudo apt-get update \
# && sudo apt-get upgrade \
# && sudo rm -rf /var/lib/apt/lists/*

COPY ./ ./

RUN pip3 install -r requirements_docker.txt

EXPOSE 8888