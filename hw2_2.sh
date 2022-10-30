#!/bin/bash

# TODO - run your inference Python3 code
mkdir $1
python ./p2/reference.py --no-train --outpath $1 --model ./p2/ckpt/model_99_4.pth