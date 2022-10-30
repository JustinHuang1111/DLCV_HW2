import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dataset import Infdataset, InfUSPSdataset
from model import DANN


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--datapath", "-d", type=str, default="./", help="path to images"
    )
    parser.add_argument(
        "--outpath", "-o", type=str, default="./test_pred.csv", help="where to save csv"
    )
    return parser.parse_args()


def test(datapath, outpath, ckptpath):
    target_data_path = datapath

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    alpha = 0

    # valid set -> without labels
    if "svhn" in target_data_path:
        target_valid_set = Infdataset(target_data_path)
    elif "usps" in target_data_path:
        target_valid_set = InfUSPSdataset(target_data_path)

    target_valid_loader = DataLoader(target_valid_set, batch_size=batch_size)

    model = torch.load(ckptpath, map_location="cuda:0")
    model = model.eval()

    n_total = 0
    n_correct = 0
    valid_loss = []
    features = torch.empty((0, 50, 4, 4), dtype=torch.float32).to(device)
    labels = torch.empty(0).to(device)
    prediction = pd.DataFrame({"image_name": [], "label": []}).astype(int)
    total = 0
    for valid_image in tqdm(target_valid_loader):

        valid_image = valid_image.to(device)

        class_output, _ = model(input_data=valid_image, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1].detach().cpu()
        for i, p in enumerate(pred):
            # print(int(pred[i]))
            new_row = pd.DataFrame(
                {
                    "image_name": target_valid_set.get_filelist()[total],
                    "label": pred[i],
                },
                index=[0],
            )
            prediction = pd.concat([prediction, new_row])
            total += 1
        pred = pred.cpu()

    prediction.to_csv(outpath, index=False)


if __name__ == "__main__":
    args = get_args()
    if "svhn" in args.datapath:
        test(args.datapath, args.outpath, "./p3/ckpt/test_aug_model_210.ckpt")
    elif "usps" in args.datapath:
        test(args.datapath, args.outpath, "./p3/ckpt/usps_4_model_46.ckpt")
