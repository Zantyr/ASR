# Q client
# This is to automatically install uniform PICTEC environment, so we don't need to port everything
# across platforms.

# python 3.6
# cuda 10.0
# julia notebook

##### TENSORFLOW PART
# This is cloned from Official tensorflow image, update as needed
FROM nvidia/cuda:10.0-devel-ubuntu18.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libnccl2=2.3.7-1+cuda10.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

##### You have to supply your own tensorflow build for cuda 10
# this wheel should be located as build/tensorflow.whl in host filesystem during image building
COPY tensorflow_pkg /static/tensorflow_pkg
RUN /venv/bin/python -m pip install /static/tensorflow_pkg/tensorflow*.whl

##### Install Q server service and Q package

COPY Q /static/Q
RUN /venv/bin/python -m pip install /static/Q

##### Install nvtop

# RUN apt-get -y install cmake libncurses5-dev git
# RUN git clone https://github.com/Syllo/nvtop.git
# RUN mkdir -p nvtop/build && cd nvtop/build && cmake .. -DNVML_RETRIEVE_HEADER_ONLINE=True && make && make install

##### Get julia

RUN apt-add-repository 'deb http://archive.ubuntu.com/ubuntu/ disco main'
RUN apt-add-repository 'deb http://archive.ubuntu.com/ubuntu/ disco universe'
RUN apt-get -y update
RUN apt-get -y install --no-install-recommends julia curl
RUN julia -e 'using Pkg; Pkg.add("IJulia"); Pkg.add("Knet")'

##### Run Jupyter

EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && source /static/runtime.sh && jupyter notebook --notebook-dir=/asr --ip 0.0.0.0 --no-browser --allow-root"]
