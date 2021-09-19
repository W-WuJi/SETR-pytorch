from tast_car_seg import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from myloss import *


def GPU2CPU():
    #  将只适用于双GPU的model转存成cpu模型 以在训练的同时使用model
    device = torch.device("cuda")
    model = build_model()
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("./checkpoints/SETR_car.pkh", map_location="cuda"))

    model = model.module
    torch.save(model.state_dict(), "./checkpoints/SETR_car_cpu.pkh")


def compute_dice():
    #  重新加载
    device = torch.device("cuda")
    model = build_model()
    model.load_state_dict(torch.load("./checkpoints/SETR_car_cpu.pkh", map_location="cpu"))

    # train_img_url = img_url[:train_size]
    # train_mask_url = mask_url[:train_size]
    # val_img_url = train_img_url
    # val_mask_url = train_mask_url

    n = 0
    dice = 0
    model.eval()
    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for val_img, val_mask in tqdm(val_loader, total=len(val_loader)):
            n += 1
            val_img = val_img
            val_mask = val_mask
            pred_img = model(val_img)
            if out_channels == 1:
                pred_img = pred_img.squeeze(1)
            cur_dice = compute_dice(pred_img, val_mask)
            dice += cur_dice
        dice = dice / n
        print("mean dice is " + str(dice))
        model.train()


if __name__ == '__main__':
    GPU2CPU()
    compute_dice()
