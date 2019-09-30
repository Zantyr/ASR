#!/bin/bash

# update repos
apt-get update

# get git and sound eXchange
apt-get -y install git sox libsox-fmt-all

# install additional python modules
pip install -r requirements.txt

# download swig
apt-get install -y libeigen3-dev scons cmake libgtest-dev  # For testing.
git clone https://github.com/google/carfac.git
export EIGEN_PATH=/usr/include/eigen3
export GTEST_SOURCE=/usr/src/gtest
apt-get install -y mercurial python3-numpy-dev
hg clone https://bitbucket.org/MartinFelis/eigen3swig

# Build swig interface for carfac
swig -python -c++ carfac.i
g++ -fPIC -shared -std=c++11 carfac/cpp/sai.cc carfac/cpp/ear.cc carfac/cpp/carfac.cc -I/usr/include/eigen3  -Doverride= -o carfac/cpp/all.so
g++ -c -fPIC -std=c++11 carfac_wrap.cxx -I/usr/include/python3.5 -I/usr/include/eigen3 -o carfac_wrap.o
g++ -shared -fPIC carfac_wrap.o carfac/cpp/all.so -o _carfac.so

# install Pynini and openfst
git clone https://github.com/kylebgorman/pynini
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
tar -xzvf openfst-1.7.2.tar.gz
cd openfst-1.7.2 && ./configure --enable-grm --enable-pdf --enable-mpdt --enable-python --enable-far
cd openfst-1.7.2 && make && make install
cd pynini && python setup.py install
yes | apt install graphviz

