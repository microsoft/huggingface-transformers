#!/bin/bash -ex

rm -rf res/*

./run.sh 1 --fp16 --ortmodule
./run.sh 1 --fp16
./run.sh 1 --ortmodule
./run.sh 1

./run.sh 4 --fp16 --ortmodule
./run.sh 4 --fp16
./run.sh 4  --ortmodule
./run.sh 4

