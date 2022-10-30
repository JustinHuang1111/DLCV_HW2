import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dataset import DANNdataset, DANNdataset_for_usps
from model import DANN


def test(datapath, valid_name, epoch, ckptpath, exp_name):
    target_data_path = os.path.join(datapath, valid_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    alpha = 0

    # valid set -> without labels
    if valid_name == "svhn":
        target_valid_set = DANNdataset(target_data_path, train=False, source=False)
    else:
        target_valid_set = DANNdataset_for_usps(
            target_data_path, train=False, source=False
        )

    target_valid_loader = DataLoader(
        target_valid_set, batch_size=batch_size, shuffle=True
    )

    model = torch.load(ckptpath, map_location="cuda:0")
    model = model.eval()
    loss_class = torch.nn.NLLLoss()

    n_total = 0
    n_correct = 0
    valid_loss = []
    features = torch.empty((0, 50, 4, 4), dtype=torch.float32).to(device)
    labels = torch.empty(0).to(device)
    for valid_data in tqdm(target_valid_loader):

        valid_image, valid_label = valid_data

        valid_image = valid_image.to(device)

        class_output, _ = model(input_data=valid_image, alpha=alpha)
        loss = loss_class(class_output, valid_label.to(device))
        pred = class_output.data.max(1, keepdim=True)[1]

        pred = pred.cpu()
        valid_label = valid_label.cpu()
        n_correct += pred.eq(valid_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        valid_loss.append(loss.item())

    accu = n_correct.data.numpy() * 1.0 / n_total

    print("epoch: %d, accuracy of the %s dataset: %f" % (epoch, valid_name, accu))
    with open(f"./{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    # test(source_dataset_name, epoch)
    # test(target_dataset_name, epoch)

    return sum(valid_loss) / len(valid_loss), accu
