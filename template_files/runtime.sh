#!/bin/bash

source /venv/bin/activate
export JUPYTER_CONFIG_DIR=/root/.jupyter

apt update
apt install -y git
apt install -y sox

pushd /asr/fwks_submod
git pull
popd

pip install git+https://github.com/detly/gammatone.git
