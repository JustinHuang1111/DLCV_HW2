# Import necessary  standard packages.
import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

# This is for the progress bar.
from tqdm.auto import tqdm

# self-defined modules
from dataset import FaceDataset, UnNormalize
from model import DCGAN_Discriminator, DCGAN_Generator
from score import calculate_fid_given_paths, face_recog


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--expname", "-e", type=str, default="dcgan", help="Experiment name"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument(
        "--cuda", "-c", type=str, default="cuda:1", help="choose a cuda"
    )
    parser.add_argument(
        "--save", "-s", type=int, default=1, help="if user wants to save model"
    )
    return parser.parse_args()


def main(exp_name: str, model_path: str):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 1314520
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name
    params = {
        "batch_size": 128,  # Batch size during training.
        "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
        "nz": 100,  # Size of the Z latent vector (the input to the generator).
        "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
        "nepochs": 400,  # Number of training epochs.
        "lr": 0.0002,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "sample": 2,  # sample every n epochs
        "outpath": "./train_output",  # save .pth every n epochs
        "datapath": "../hw2_data/face/",
    }
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    dataset = FaceDataset(path=os.path.join(params["datapath"], "train"))
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    generator = DCGAN_Generator(params).to(device)
    discriminator = DCGAN_Discriminator(params).to(device)

    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
    opt_G = torch.optim.Adam(
        generator.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
    print(generator)
    print(discriminator)
    # schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min')
    # schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min')

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()
    # To denormalize inferenced images
    unorm = UnNormalize()

    best_fid = 500
    best_face_acc = 0
    G_losses = []
    D_losses = []
    n_epochs = params["nepochs"]

    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloader)
        progress_bar.set_description(f"Epoch {epoch+1}")
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, params["nz"], 1, 1)).to(device)  # ML
            # z = Variable(torch.randn(bs, params["nz"], 1, 1)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = generator(z)

            # Label
            r_label = torch.ones((bs)).to(device)
            f_label = torch.zeros((bs)).to(device)

            # Model forwarding
            r_logit = discriminator(r_imgs.detach())
            f_logit = discriminator(f_imgs.detach())

            # Compute the loss for the discriminator.
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # Model backwarding
            discriminator.zero_grad()
            loss_D.backward()
            opt_D.step()
            # ============================================
            #  Train G
            # ============================================
            z = Variable(torch.randn(bs, params["nz"], 1, 1)).to(device)  # ML
            # z = Variable(torch.randn(bs, params["nz"], 1, 1)).to(device)
            f_imgs = generator(z)

            # Model forwarding
            f_logit = discriminator(f_imgs)
            loss_G = criterion(f_logit, r_label)

            # Model backwarding
            generator.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Save the losses for printing.
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
        # ------------------- an epoch finish ---------------------#
        G_loss = sum(G_losses) / len(G_losses)
        D_loss = sum(D_losses) / len(D_losses)

        # ------------------- generate 1000 images and evaluate---------------------#
        n_outputs = 1000
        if epoch % params["sample"] == 0:
            generator.eval()
            noise = torch.randn(n_outputs, params["nz"], 1, 1, device=device)  # ML
            # noise = torch.randn(n_outputs, params["nz"], 1, 1, device=device)
            imgs = generator(noise).detach().cpu()
            for i in range(n_outputs):
                vutils.save_image(
                    unorm(imgs[i]), os.path.join(params["outpath"], f"{i}.png")
                )
            # ------------------- get score ---------------------#
            print("Calculating FID...")
            fid = calculate_fid_given_paths(
                [os.path.join(params["datapath"], "val"), params["outpath"]],
                params["batch_size"],
                device,
                2048,
                0,
            )
            face_acc = face_recog(params["outpath"])

            # schedulerD.step(-1 * face_acc)
            # schedulerG.step(fid)
            print("FID: %f" % (fid))
            print("Face Accuracy: %.2f" % (face_acc))

            if fid < best_fid:
                print("Saved model with improved FID: %f -> %f" % (best_fid, fid))
                best_fid = fid
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizerG": opt_G.state_dict(),
                        "optimizerD": opt_D.state_dict(),
                        "params": params,
                    },
                    f"./checkpoints/fid_{train_name}.pth",
                )

            if face_acc > best_face_acc:
                best_face_acc = face_acc
                if fid < 27:
                    print(
                        "Saved model with improved face_acc: %f -> %f"
                        % (best_face_acc, face_acc)
                    )
                    torch.save(
                        {
                            "generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            "optimizerG": opt_G.state_dict(),
                            "optimizerD": opt_D.state_dict(),
                            "params": params,
                        },
                        f"./checkpoints/face_{train_name}.pth",
                    )

        with open(f"./logs/{train_name}_log.txt", "a") as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f} , best_score = ({best_fid:.3f} , {best_face_acc:.3f})\n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f} , best_score = ({best_fid:.3f} , {best_face_acc:.3f})"
            )


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.model)
