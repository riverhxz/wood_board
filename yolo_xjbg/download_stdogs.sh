#!/usr/bin/env bash

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar xf images.tar
tar xf annotation.tar
mkdir data
mv Images Annotation data