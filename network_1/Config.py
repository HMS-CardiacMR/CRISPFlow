# Manuel A. Morales (mmoraleq@bidmc.harvard.edu)
# Beth Israel Deaconess Medical Center
# Harvard Medical School

import os
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from Models import *
import sys

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(8888)
cudnn.benchmark = True
cudnn.deterministic = False

use_gpu=True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0")

mode = "train"

isPreTrained=False

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":


    # 1. Dataset path.
    coderoot = '/mnt/alp/Users/Manuel/F01Code/Pythoncodes/1_Flow/Training/002_flow_model_2'
    dataroot = "/data2/manuel/1_REGAIN_Flow/dtrn"
    
    image_size=image_In

    upscale_factor=scale_factor
    batch_size = 16

    # 2. Define model.
    generator = GeneratorRRDB(1,filters=64, num_res_blocks=23).to(device)
    discriminator = Discriminator(input_shape=(1, image_size, image_size)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # 3. Resume training.
    seed=123
    start_p_epoch = 0
    start_g_epoch = 0

    # 4. Number of epochs.
    g_epochs = 40000

    warmup_batches=500

    # 5. Loss function.
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_FFT = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.MSELoss().to(device)

    # Loss function weight.
    lambda_adv = 5e-03
    lambda_pixel = 1e-02
    lambda_fft = 1e-02

    # 6. Optimizer.
    p_lr = 1e-4
    d_lr = 1e-4
    g_lr = 1e-4

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    os.makedirs(os.path.join(coderoot, "samples"), exist_ok=True)