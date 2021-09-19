# from tast_car_seg import *
import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms as transforms


def seg_channel21():
    """
     用于将三通道segmentation转换为单通道01二值图
    """

    mask_url = sorted(glob.glob("./segmentation_Crack/imgs/*"))
    for i in range(len(mask_url)):
        mask = Image.open(mask_url[i])
        mask = np.array(mask, dtype=np.uint0)
        if mask.shape[-1] != 3:
            # mask = mask[..., 0]  # masks：{ndarray：(1080,1920,3)}   选取Red通道判断像素值(大概不适用于无red标注)
            print(mask_url[i])
        # mask = mask / 128
        plt.imsave("./segmentation_Crack/masks_binary/%s" % mask_url[i][27:-4] + '.gif', mask, cmap="binary_r")


class DataAugmentation:
    """
    用于图像增强
    """

    def __init__(self, img, mask):
        self.img = img
        self.mask = mask

    def flip(self):
        # 翻转
        hor_im = transforms.RandomHorizontalFlip(p=1)(self.img)  # p表示概率
        ver_im = transforms.RandomVerticalFlip(p=1)(self.img)
        hor_mask = transforms.RandomHorizontalFlip(p=1)(self.mask)
        ver_mask = transforms.RandomVerticalFlip(p=1)(self.mask)
        return hor_im, ver_im, hor_mask, ver_mask

    def rotation(self):
        # 旋转
        angle = transforms.RandomRotation.get_params([-180, 180])  # -180~180随机选一个角度旋转
        rot_img = self.img.rotate(angle)
        rot_mask = self.mask.rotate(angle)
        # rot_mask = transforms.RandomRotation(45)(self.mask)    #随机旋转45度
        return rot_img, rot_mask

    def crop(self):
        # 随即剪裁
        location = transforms.RandomCrop.get_params(self.img, output_size=(700, 1200))
        location = list(location)
        location[0], location[1], location[2], location[3] = location[1], location[0], location[3], location[2]
        location[2] += location[0]
        location[3] += location[1]
        crop_img = self.img.crop(location)
        crop_mask = self.mask.crop(location)
        return crop_img, crop_mask

    def bright(self):
        # 随机亮度
        new_im = transforms.ColorJitter(brightness=[0.5, 2.0])(self.img)
        new_mask = self.mask
        return new_im, new_mask


def data_Augmentation():
    img_url = sorted(glob.glob("./segmentation_Crack/imgs/*"))
    mask_url = sorted(glob.glob("./segmentation_Crack/masks_binary/*"))
    for i in range(len(img_url)):
        img = Image.open(img_url[i])
        mask = Image.open(mask_url[i])
        DA = DataAugmentation(img, mask)
        hor_im, ver_im, hor_mask, ver_mask = DA.flip()
        hor_im.save(img_url[i][0:-4] + "_HorizontalFlip.jpg")
        ver_im.save(img_url[i][0:-4] + "_VerticalFlip.jpg")
        hor_mask.save(mask_url[i][0:-4] + "_HorizontalFlip.gif")
        ver_mask.save(mask_url[i][0:-4] + "_VerticalFlip.gif")

    img_url = sorted(glob.glob("./segmentation_Crack/imgs/*"))
    mask_url = sorted(glob.glob("./segmentation_Crack/masks_binary/*"))
    for i in range(len(img_url)):
        img = Image.open(img_url[i])
        mask = Image.open(mask_url[i])
        DA = DataAugmentation(img, mask)
        new_im, new_mask = DA.bright()
        new_im.save(img_url[i][0:-4] + "_Bright.jpg")
        new_mask.save(mask_url[i][0:-4] + "_Bright.gif")

    img_url = sorted(glob.glob("./segmentation_Crack/imgs/*"))
    mask_url = sorted(glob.glob("./segmentation_Crack/masks_binary/*"))
    for i in range(len(img_url)):
        img = Image.open(img_url[i])
        mask = Image.open(mask_url[i])
        DA = DataAugmentation(img, mask)
        crop_img, crop_mask = DA.crop()
        crop_img.save(img_url[i][0:-4] + "_Crop.jpg")
        crop_mask.save(mask_url[i][0:-4] + "_Crop.gif")


if __name__ == '__main__':
    data_Augmentation()
    # seg_channel21()
    # predict()
    # model = build_model()
    # print(model)
    # total = sum(p.numel() for p in model.parameters())

    # print("Total params: %.2fM" % (total / 1e6))
