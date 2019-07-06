# Q client
# This is to automatically install uniform PICTEC environment, so we don't need to port everything
# across platforms.

# Specifications:
# Python 3.5
# cuda 9
# ubuntu 16.04


##### TENSORFLOW PART
FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update && apt-get upgrade -y python3

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
        apt-get update && \
        apt-get install libnvinfer4=4.1.2-1+cuda9.0

ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools


##### Install dependencies
RUN apt-get install -y --no-install-recommends python${_PY_SUFFIX}-virtualenv
RUN python3 -m virtualenv --python python3 /venv

RUN /venv/bin/python -m pip install jupyter
VOLUME /asr
COPY requirements.txt /static/requirements.txt
COPY runtime.sh /static/runtime.sh
RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter
RUN /venv/bin/python -m pip install -r /static/requirements.txt

##### Install tensorflow
ARG TF_PACKAGE=tensorflow-gpu
RUN /venv/bin/python -m pip install ${TF_PACKAGE}

##### Install Q server service and Q package
COPY Q /static/Q
RUN /venv/bin/python -m pip install /static/Q

##### Dumb init
RUN apt-get install -y --no-install-recommends wget
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.2/dumb-init_1.2.2_amd64.deb
RUN dpkg -i dumb-init_*.deb

##### Run Jupyter
EXPOSE 8888
EXPOSE 8900
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["bash", "-c", "source /etc/bash.bashrc && source /static/runtime.sh && exec env LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH jupyter notebook --notebook-dir=/asr --ip 0.0.0.0 --no-browser --allow-root"]
