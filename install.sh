#!/bin/bash

# update repos
apt-get update

# get git and sound eXchange
apt-get -y install git sox libsox-fmt-all python3-tk graphviz

# install additional python modules
pip install -r requirements.txt

# install Pynini and openfst
git clone https://github.com/kylebgorman/pynini
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
tar -xzvf openfst-1.7.2.tar.gz
pushd openfst-1.7.2

# apply patch openfst to make it installable
cat configure | sed 's/print ver\(.*\)"/print(ver \1)"/g' > /tmp/conf_script
cat /tmp/conf_script | sed 's/string\.split/str.split/g' > configure

./configure --enable-grm --enable-pdf --enable-mpdt --enable-python --enable-far
make
make install
popd
pushd pynini
python setup.py install
popd

# update fwks library
# git clone https://github.com/Zantyr/fwks
