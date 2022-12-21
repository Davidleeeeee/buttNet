import numpy as np
from PIL import Image
import argparse
import os
import os.path as osp
import glob
import random


# 沿着X翻转
def flip_x(image):
    image = np.asarray(image)
    image1 = np.flipud(image)
    return Image.fromarray(image1)


# 沿着Y翻转
def flip_y(image):
    image = np.asarray(image)
    image1 = np.fliplr(image)
    return Image.fromarray(image1)


# 沿着对角线翻转
def flip45(image):
    image = np.asarray(image)
    img1 = np.transpose(image, (1, 0, 2))
    return Image.fromarray(flip_x(flip_y(img1)))


# 旋转 (rot90)
def rot90(image):
    image = np.asarray(image)
    img1 = np.rot90(image, 1)
    return Image.fromarray(img1)


def resize(image, size):
    size = size.split(',')
    image = image.resize((int(size[0]), int(size[1])))
    return image


def resize16(image):
    w, h = image.size
    new_w = w - w % 16
    new_h = h - h % 16
    image = image.resize((int(new_w), int(new_h)))
    return image


def patches(image, rate):
    w, h = image.size
    img1 = np.asarray(image)
    new_w = w//10
    new_h = h//10
    cards = [i for i in range(100)]
    random.shuffle(cards)
    for i in range(rate):
        x = cards[i] % 10
        y = cards[i] // 10
        # 对每个坐标区域颜色修改-背景色
        RGB_data = [255, 255, 255]
        for i_y in range(new_h):
            for i_x in range(new_w):
                img1[i_y+new_h*y][i_x+new_w*x] = RGB_data
    return Image.fromarray(img1)


def crop(image):
    box = (15, 0, 151, 130)
    region = image.crop(box)
    return region


def switch(image, string, size):
    if string == 'flip_x':
        return flip_x(image)
    elif string == 'flip_y':
        return flip_y(image)
    elif string == 'flip45':
        return flip45(image)
    elif string == 'resize':
        return resize(image, size)
    elif string == 'resize16':
        return resize16(image)
    elif string == 'crop':
        return crop(image)
    else:
        return rot90(image)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', default='/home/mento_algo/文档/picture_error/异物crop', help='input train data directory')
    parser.add_argument('--output_dir', default='/home/mento_algo/文档/picture_error/异物crop2', help='output train data directory')
    parser.add_argument('--size', default=[0, 0], help='string, which process do you want')
    parser.add_argument('--type', default='crop', help='string, which process do you want')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 初始化参数
    input_dir = args.input_dir
    output_target_dir = args.output_dir
    process_type = args.type
    size = args.size
    # 判断路径否存在
    if not osp.exists(output_target_dir):
        os.makedirs(output_target_dir)
        print('Creating train data directory:', output_target_dir)

    # label_set = np.array([0])
    for file_name in glob.glob(osp.join(input_dir, '*.bmp')):
        img_out = switch(Image.open(file_name), process_type, size)
        base = osp.splitext(osp.basename(file_name))[0]
        out_bmp_file = osp.join(output_target_dir, base + process_type + '.bmp')
        img_pil = img_out
        img_pil.save(out_bmp_file)

        #unin
        # img = np.array(Image.open(file_name))
        # label_set = np.union1d(label_set, np.unique(img))
    # print(label_set)
