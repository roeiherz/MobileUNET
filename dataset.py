import random
import re
from glob import glob

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from config import IMG_DIR


def _mask_to_img(mask_file):
    img_file = re.sub('^{}/masks'.format(IMG_DIR), '{}/images'.format(IMG_DIR), mask_file)
    img_file = re.sub('\.ppm$', '.jpg', img_file)
    return img_file


def _img_to_mask(img_file):
    mask_file = re.sub('^{}/images'.format(IMG_DIR), '{}/masks'.format(IMG_DIR), img_file)
    # mask_file = re.sub('\.jpg$', '.ppm', mask_file)
    return mask_file


def get_img_files_eval():
    mask_files = sorted(glob('{}/masks/*.jpg'.format(IMG_DIR)))
    return np.array([_mask_to_img(f) for f in mask_files])


def get_img_files():
    mask_files = sorted(glob('{}/masks/*.jpg'.format(IMG_DIR)))
    # mask_files = mask_files[:10000]
    sorted_mask_files = []

    # Sorting out
    for msk in mask_files:
        # Sort out black masks
        msk_img = cv2.imread(msk)
        if len(np.where(msk_img == 1)[0]) == 0:
            continue

        # Sort out night images
        img_path = re.sub('^{}/masks'.format(IMG_DIR), '{}/images'.format(IMG_DIR), msk)
        img = cv2.imread(img_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        higher_img = gray_image[0:120, :]
        if np.average(higher_img) > 100:
            # Day image, so append
            sorted_mask_files.append(msk)

    # return np.array([_mask_to_img(f) for f in mask_files])
    return np.array([_mask_to_img(f) for f in sorted_mask_files])


class MaskDataset(Dataset):
    def __init__(self, img_files, transform, mask_transform=None, mask_axis=0):
        self.img_files = img_files
        self.mask_files = [_img_to_mask(f) for f in img_files]
        self.transform = transform
        if mask_transform is None:
            self.mask_transform = transform
        else:
            self.mask_transform = mask_transform
        self.mask_axis = mask_axis

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, self.mask_axis]

        seed = random.randint(0, 2 ** 32)

        # Apply transform to img
        random.seed(seed)
        img = Image.fromarray(img)
        img = self.transform(img)

        # Apply same transform to mask
        random.seed(seed)
        mask = Image.fromarray(mask)
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    pass
    #
    # mask = cv2.imread('{}/masks/Aaron_Peirsol_0001.ppm'.format(IMG_DIR))
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # mask = mask[:, :, 0]
    # print(mask.shape)
    # plt.imshow(mask)
    # plt.show()
