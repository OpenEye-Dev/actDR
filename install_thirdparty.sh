#!/bin/bash
# this is just for testing and iterating
mkdir data
mkdir output
mkdir output/log
mkdir output/checkpoints
rm -rf thirdparty
mkdir thirdparty
cd thirdparty
git clone https://github.com/tswedish/jeepers.git
cd jeepers
./install_thirdparty.sh
