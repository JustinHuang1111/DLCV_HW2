#!/bin/bash

# TODO - run your inference Python3 code
mkdir $2
python ./p3/inference.py --datapath $1 --outpath $2