FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	netbase \
	wget \
	git \
	openssh-client \
	ssh \
	vim \
 && rm -rf /var/lib/apt/lists/*

# http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# https://github.com/docker-library/python/issues/147
ENV PYTHONIOENCODING UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.6 \
	python3.6-dev \
	python3-pip \
	python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /duke-dbt-detection

RUN pip3 install http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
 && pip3 install torchvision==0.2.1

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt
