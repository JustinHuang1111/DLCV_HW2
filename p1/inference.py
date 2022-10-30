import argparse
import os
import random

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

from dataset import UnNormalize
from model import DCGAN_Generator, DCGAN_Generator_ML, WGANGP_Generator

parser = argparse.ArgumentParser()
parser.add_argument("--model1", "-m", type=str, default=None, help="Model path to load")
parser.add_argument("--model2", "-n", type=str, default=None, help="Model path to load")
# parser.add_argument("-model3", "-v", type=str, default=None, help="Model path to load")
parser.add_argument("--seed", "-s", type=int, default=1, help="select seed")
parser.add_argument("--outpath", "-o", type=str, default="./output", help="output path")
parser.add_argument("--DCGAN", action="store_true")
parser.add_argument("--WGANGP", dest="DCGAN", action="store_false")
parser.set_defaults(DCGAN=True)
parser.add_argument("--cuda", "-c", default="cuda", help="choose a cuda")
args = parser.parse_args()
n_outputs = 1000

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)
device = torch.device(args.cuda)

if args.DCGAN:
    state_dict = torch.load(args.model1)
    params = state_dict["params"]
    generator1 = DCGAN_Generator(params).to(device)
    generator1.load_state_dict(state_dict["generator"])
    generator1.eval()

    state_dict = torch.load(args.model2)
    params = state_dict["params"]
    generator2 = DCGAN_Generator_ML().to(device)
    generator2.load_state_dict(state_dict["generator"])
    generator2.eval()

else:
    state_dict = torch.load(args.model1)
    params = state_dict["params"]
    generator = WGANGP_Generator(params).to(device)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()

unorm = UnNormalize()

input1 = Variable(torch.randn(n_outputs, params["nz"], 1, 1)).to(device)
input2 = Variable(torch.randn(n_outputs, params["nz"])).to(device)

inference = torch.empty((n_outputs, 3, 64, 64))
with torch.no_grad():
    if args.DCGAN:
        imgs1 = generator1(input1).detach().cpu()
        imgs2 = generator2(input2).detach().cpu()
        for i in range(n_outputs):
            if random.random() > 0.44:
                inference[i] = unorm(imgs1[i])
            else:

                inference[i] = unorm(imgs2[i])
            vutils.save_image(inference[i], os.path.join(args.outpath, f"{i}.png"))

    else:
        imgs = generator(input1).detach().cpu()
        for i in range(n_outputs):
            inference[i] = unorm(imgs[i])

            vutils.save_image(inference[i], os.path.join(args.outpath, f"{i}.png"))

    # filename = os.path.join(
    #     ".",
    #     "report_32_wgan.png",
    # )
    # vutils.save_image(inference_imgs[:32], filename, nrow=8)
