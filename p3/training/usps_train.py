# Reference:
# https://github.com/NaJaeMin92/pytorch_DANN
# https://github.com/fungtion/DANN
from test import test

from train_on_source import trainOnSource
from train_on_target import trainOnTarget

from train import train

# train("../hw2_data/digits", "mnistm", "usps", "./ckpt", "usps_final")
# trainOnSource("../hw2_data/digits", "mnistm", "usps", "./ckpt_report", "onSource_usps")
trainOnTarget("../hw2_data/digits", "mnistm", "usps", "./ckpt_report", "onTargetUSPS")

# test("../hw2_data/digits", "svhn", 0, "./ckpt/test_model_0.ckpt")
