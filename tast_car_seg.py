# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
import csv
import glob
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from SETR.transformer_seg import SETRModel
from myloss import SoftDiceLoss

ImageFile.LOAD_TRUNCATED_IMAGES = True
img_url = sorted(glob.glob("./segmentation_Crack/imgs/*"))
mask_url = sorted(glob.glob("./segmentation_Crack/masks_binary/*"))
seed = 50
random.seed(seed)
random.shuffle(img_url)
random.seed(seed)
random.shuffle(mask_url)

train_size = int(len(img_url) * 0.8)
train_img_url = img_url[:train_size]
train_mask_url = mask_url[:train_size]
val_img_url = img_url[train_size:]
val_mask_url = mask_url[train_size:]
log_dir = "./checkpoints/SETR_car.pkh"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
epoches = 10000
out_channels = 1


def build_model():
    model = SETRModel(patch_size=(16, 16),
                      in_channels=3,
                      out_channels=1,
                      hidden_size=1024,
                      num_hidden_layers=6,
                      num_attention_heads=8,
                      decode_features=[512, 256, 128, 64])
    return model


class CarDataset(Dataset):
    def __init__(self, img_url, mask_url):
        super(CarDataset, self).__init__()
        self.img_url = img_url
        self.mask_url = mask_url

    def __getitem__(self, idx):
        img = Image.open(self.img_url[idx]).convert('RGB')  # 一些数据图片已经被转化为灰度图
        img = img.resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255
        mask = Image.open(self.mask_url[idx])
        mask = mask.resize((256, 256))
        mask = np.array(mask, dtype=np.float32)
        img_array = img_array.transpose(2, 0, 1)

        return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

    def __len__(self):
        return len(self.img_url)


def compute_dice(input, target):
    eps = 0.0001

    input = (input > 0.5).float()  # input 是经过了sigmoid之后的输出。
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t


def predict():
    # device = torch.device("cpu")
    model = build_model()
    # model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(
        torch.load("./checkpoints/202106201909/SETR_crack.pkh", map_location="cpu"))
    print(model)
    # val_dataset = CarDataset(val_img_url, val_mask_url)
    val_dataset = CarDataset(["./segmentation_Crack/20201023-173322-147.jpg"],
                             ["./segmentation_Crack/masks_binary/20200511-115256-160.gif"])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for img, mask in val_loader:
            # img = img.to(device)
            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).int()
            plt.subplot(1, 3, 1)
            print(img.shape)
            img = img.permute(0, 2, 3, 1)
            plt.imshow(img[0].cpu())  # tensor转array操作必须在cpu上
            plt.subplot(1, 3, 2)
            plt.imshow(pred[0].cpu().squeeze(0), cmap="gray")
            # plt.subplot(1, 3, 3)
            # plt.imshow(mask[0], cmap="gray")
            plt.show()


def train():
    # 储存当前模型并创建log.csv
    train_time = time.strftime("%Y%m%d%H%M", time.localtime())
    checkpoint_path = f"./checkpoints/{train_time}"
    os.makedirs(checkpoint_path)
    shutil.copyfile("./tast_car_seg.py", checkpoint_path + "/tast_car_seg.py")
    shutil.copyfile("./SETR/transformer_seg.py", checkpoint_path + "/transformer_seg.py")
    shutil.copyfile("./SETR/transformer_model.py", checkpoint_path + "/transformer_model.py")
    with open(checkpoint_path + "/log.csv", 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["epoch", "epoch_loss", "mean dice"])
    model = build_model()
    # model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU 会重启
    model.to(device)

    train_dataset = CarDataset(train_img_url, train_mask_url)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

    val_dataset = CarDataset(val_img_url, val_mask_url)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loss_func = SoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)

    step = 0
    report_loss = 0.0
    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     model.load_state_dict(checkpoint)
    #     # model.load_state_dict(checkpoint['model'])
    #     # optimizer.load_state_dict(checkpoint['optimizer'])
    #     # start_epoch = checkpoint['epoch']
    #     start_epoch = 0
    #     print('加载 epoch {} 成功！'.format(start_epoch))
    # else:
    #     start_epoch = 0
    #     print('无保存模型，将从头开始训练！')
    for epoch in range(0, epoches):
        print("epoch is " + str(epoch))
        for img, mask in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()  # 清空过往梯度
            step += 1
            img = img.to(device)
            mask = mask.to(device)

            pred_img = model(img)  # pred_img (batch, len, channel, W, H)
            if out_channels == 1:
                pred_img = pred_img.squeeze(1)  # 去掉通道维度

            loss = loss_func(pred_img, mask)
            report_loss += loss.item()  # 提取loss的数值
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新网络参数

            if step % 1000 == 0:  # 每1000个step(batch?)算一组val mean dice + 更新储存的模型SETR_crack.pkh
                dice = 0.0
                n = 0
                model.eval()
                with torch.no_grad():
                    print("report_loss is " + str(report_loss))  # 每1000次的loss进行加和输出
                    report_loss = 0.0

                    # 加载验证集数据计算dice
                    for val_img, val_mask in tqdm(val_loader, total=len(val_loader)):
                        n += 1
                        val_img = val_img.to(device)
                        val_mask = val_mask.to(device)
                        pred_img = model(val_img)
                        if out_channels == 1:
                            pred_img = pred_img.squeeze(1)
                        cur_dice = compute_dice(pred_img, val_mask)
                        dice += cur_dice
                    dice = dice / n
                    print("mean dice is " + str(dice))
                    torch.save(model.state_dict(), checkpoint_path + "/SETR_crack.pkh")
                    model.train()

            with open(checkpoint_path + "/log.csv", 'a', newline="") as f:
                f_csv = csv.writer(f)
                f_csv.writerow([epoch, loss, dice])


if __name__ == "__main__":
    train()
    # predict()
