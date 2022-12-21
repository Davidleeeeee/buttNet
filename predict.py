import argparse
import os.path as osp
import glob
import numpy as np
import torch
from PIL import Image
import cv2 as cv
from utils.data_loading import BasicDataset
from butterfly_net import ButterflyNet
from utils.utils import plot_five
from img_expansion import resize16
from pathlib import Path


def predict_img(net,
                full_img,
                device):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, is_mask=False, need_patch=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output1, output2 = net(img)
        return output1.cpu().permute(0, 2, 3, 1).numpy()[0], output2.cpu().permute(0, 2, 3, 1).numpy()[0]


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--load', '-f', type=str, default='./checkpoints/ok4_7Patches_epoch7.pth',
                        help='Load model from a .pth file')
    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    point = 1 / max(np.unique(mask))
    mask1 = (mask * 255 * point).astype(np.uint8)
    return Image.fromarray(mask1)


def dilate(mask: np.ndarray):
    # mask1 = (mask * 255).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.dilate(mask, kernel)
    return dst


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    pth = args.load
    net = ButterflyNet(n_classes=3, layer_numbers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(pth, map_location=device))

    image_packages = []
    count = 0
    for filename in glob.glob(osp.join(in_files[0], '*.bmp')):
        count = count + 1
        print(count)
        img = Image.open(filename)
        img = resize16(img)

        gauss_img = np.asarray(img).astype(float)
        # gauss_img = cv.GaussianBlur(gauss_img, (5, 5), 0)

        net.index = 0
        mask_img, mask_img_tow = predict_img(net=net,
                                             full_img=img,
                                             device=device)
        result_ori = np.absolute(mask_img - gauss_img / 255)
        result_origen = mask_to_image(result_ori)

        result_ct = np.absolute(mask_img_tow - mask_img)
        result = mask_to_image(result_ct)

        mask = mask_to_image(mask_img)
        mask_tow = mask_to_image(mask_img_tow)

        base = osp.splitext(osp.basename(filename))[0]

        dilate_img = dilate(result_ct)
        five_img = result_ori - dilate_img
        dilate_img = mask_to_image(dilate_img)
        five_img[five_img[::] < 0] = 0
        five_img = mask_to_image(five_img)
        image_packages.append([img, mask, result_origen, result, dilate_img, five_img, base])
    plot_five(image_packages)
