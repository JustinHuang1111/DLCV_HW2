# Reference:
# https://github.com/NaJaeMin92/pytorch_DANN
# https://github.com/fungtion/DANN
from test import test

from train_on_source import trainOnSource
from train_on_target import trainOnTarget

from train import train

# train("../hw2_data/digits", "mnistm", "svhn", "./ckpt", "svhn_final")
# trainOnSource("../hw2_data/digits", "mnistm", "svhn", "./ckpt_report", "onSource")
# trainOnTarget("../hw2_data/digits", "mnistm", "svhn", "./ckpt_report", "onTargetSVHN")
test("../hw2_data/digits", "usps", 0, "./ckpt/usps_4_model_46.ckpt", 0)  # 0.8177
# test("../hw2_data/digits", "svhn", 0,"./ckpt/test_aug_model_210.ckpt", 0) # 0.54368
# test("../hw2_data/digits", "svhn", 0,"./test_model_best.ckpt", 0) # 0.4


# test("../hw2_data/digits", "svhn", 0, "./ckpt_report/onSource_model_1.ckpt", 0) # 0.376750
# test("../hw2_data/digits", "usps", 0, "./ckpt_report/onSource_usps_model_0.ckpt", 0) # 0.728516
# test("../hw2_data/digits", "svhn", 0, "./ckpt_report/onTargetSVHN_model_278.ckpt", 0) # 0.932000
# test("../hw2_data/digits", "usps", 0, "./ckpt_report/onTargetUSPS_model_temp.ckpt", 0) # 0.958984
