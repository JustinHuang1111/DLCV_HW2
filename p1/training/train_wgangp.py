# https://www.kaggle.com/code/b09901104/dlcvhw2-problem1/edit
# Import necessary  standard packages.
import argparse
import json
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
from torchvision.utils import save_image

# This is for the progress bar.
from tqdm.auto import tqdm

# self-defined modules
from dataset import FaceDataset, UnNormalize
from gp import gp
from model import WGANGP_Discriminator, WGANGP_Generator
from score import calculate_fid_given_paths, face_recog


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--expname", "-e", type=str, default="wgan", help="Experiment name"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    parser.add_argument(
        "--save", "-s", type=int, default=1, help="determine if want to save model"
    )
    return parser.parse_args()


def main(exp_name: str, model_path: str, cuda: str, to_save: bool):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 1314520
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name

    # load params
    params = {}
    exp_path = "./exp/wgan/test6.json"
    with open(exp_path) as f:
        params = json.load(f)
    with open(exp_path, "w") as f:
        params["time"] = time
        json.dump(params, f)
    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Get the data.
    data_dir = "../hw2_data/face/"
    dataset = FaceDataset(path=os.path.join(data_dir, "train"))
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    if params["model_path"]:
        state_dict = torch.load(params["model_path"])
        # params = state_dict["params"]
        netG = WGANGP_Generator(params).to(device)
        netG.load_state_dict(state_dict["generator"])
        netD = WGANGP_Discriminator(params["nc"]).to(device)
        netD.load_state_dict(state_dict["discriminator"])
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=params["lr"])
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=params["lr"])
        optimizerD.load_state_dict(state_dict["optimizerD"])
        optimizerG.load_state_dict(state_dict["optimizerG"])

    else:
        netG = WGANGP_Generator(params).to(device)
        netD = WGANGP_Discriminator(params["nc"]).to(device)

        # change the optimizer to RMSprop
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=params["lr"])
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=params["lr"])

    # test scheduler
    if params["sch"]:
        schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, "min")
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, "min")
    print(netG)
    print(netD)
    # Binary Cross Entropy loss function.
    G_losses = []
    D_losses = []
    n_epochs = params["nepochs"]
    Tensor = torch.FloatTensor
    unorm = UnNormalize()
    best_face_acc = 0
    best_fid = 500
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloader)
        progress_bar.set_description(f"Epoch {epoch+1}")
        for i, imgs in enumerate(progress_bar):
            bs = imgs.size(0)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            netD.zero_grad()
            # Sample noise as generator input
            z = Variable(torch.randn(bs, params["nz"], 1, 1)).to(device)

            fake_imgs = netG(z)
            r_logit = netD(real_imgs)
            f_logit = netD(fake_imgs)
            gradient_penalty = gp(netD, real_imgs.data, fake_imgs.data)
            print(
                -torch.mean(r_logit).item(),
                torch.mean(f_logit).item(),
                params["lambda_gp"] * gradient_penalty.item(),
            )
            d_loss = (
                -torch.mean(r_logit)
                + torch.mean(f_logit)
                + params["lambda_gp"] * gradient_penalty
            )

            d_loss.backward()
            optimizerD.step()

            # wgan clipvalue
            for p in netD.parameters():
                p.data.clamp_(-params["clip_value"], params["clip_value"])

            D_losses.append(d_loss.item())
            netG.zero_grad()
            # Train the generator every n_critic steps
            # -----------------
            #  Train Generator
            # -----------------
            if i % params["n_critic"] == 0:
                fake_imgs = netG(z)
                # Loss measures generator's ability to fool the discriminator
                f_logit = netD(fake_imgs)
                g_loss = -torch.mean(f_logit)
                g_loss.backward()
                optimizerG.step()
                G_losses.append(g_loss.item())

        D_loss = sum(D_losses) / len(D_losses)
        G_loss = sum(G_losses) / len(G_losses)

        # Save the model.
        if to_save:
            torch.save(
                {
                    "generator": netG.state_dict(),
                    "discriminator": netD.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "params": params,
                },
                f"./checkpoints/{train_name}.pth",
            )
        # ------------------- generate 1000 images and evaluate---------------------#
        n_outputs = 1000
        if epoch % params["sample_interval"] == 0:
            netG.eval()
            noise = torch.randn(n_outputs, params["nz"], 1, 1, device=device)
            imgs = netG(noise).detach().cpu()
            for i in range(n_outputs):
                vutils.save_image(
                    unorm(imgs[i]), os.path.join(params["output_path"], f"{i}.png")
                )
            # ------------------- get score ---------------------#
            print("Calculating FID...")
            fid = calculate_fid_given_paths(
                [os.path.join(params["data_dir"], "val"), params["output_path"]],
                params["batch_size"],
                device,
                2048,
                0,
            )
            face_acc = face_recog(params["output_path"])

            # scheduler update
            if params["sch"]:
                schedulerD.step(-1 * face_acc)
                schedulerG.step(fid)

            print("FID: %f" % (fid))
            print("Face Accuracy: %.2f" % (face_acc))

            # ------------------- save model depending on score ---------------------#
            if fid <= 27 and face_acc >= 90:
                print("Saved model with Success: %f -> %f" % (fid, face_acc))
                torch.save(
                    {
                        "generator": netG.state_dict(),
                        "discriminator": netD.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
                        "params": params,
                    },
                    f"./checkpoints/pass_{train_name}.pth",
                )

            if fid < best_fid:
                print("Saved model with improved FID: %f -> %f" % (best_fid, fid))
                best_fid = fid
                torch.save(
                    {
                        "generator": netG.state_dict(),
                        "discriminator": netD.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
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
                            "generator": netG.state_dict(),
                            "discriminator": netD.state_dict(),
                            "optimizerG": optimizerG.state_dict(),
                            "optimizerD": optimizerD.state_dict(),
                            "params": params,
                        },
                        f"./checkpoints/acc_{train_name}.pth",
                    )

        with open(f"./logs/{train_name}_log.txt", "a") as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f} , best_score = ({best_fid:.3f} , {best_face_acc:.3f})\n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f} , best_score = ({best_fid:.3f} , {best_face_acc:.3f})"
            )

        with open(exp_path, "w") as f:
            params["result"] = f"best_score = ({best_fid:.3f} , {best_face_acc:.3f})"
            json.dump(params, f)


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.model, args.cuda, args.save)
