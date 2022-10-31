#!/bin/bash

# TODO - run your inference Python3 code
python ./p1/inference.py --model1 ./p1/ckpt/1029-0209_dcgan.pth --model2 ./p1/ckpt/1028-0847_dcgan.pth --seed 99 --outpath $1
