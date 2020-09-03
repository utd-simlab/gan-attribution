#!/usr/bin/env python3

import argparse
import datetime
import logging
import math
import os
import json
import sys

from tqdm import tqdm
import numpy as np
import skimage
import skimage.transform
import webp
from PIL import Image
import imageio
import scipy

def open_logger(logfile, level):
    logger = logging.getLogger('gan_detection')
    #File handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #Stream handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    logger.setLevel(level)
    return logger

def make_dir(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def load_json(pth):
    with open(pth, 'r') as fh:
        obj = json.load(fh)
    return obj

def save_json(obj, pth):
    with open(pth, 'w') as fh:
        json.dump(obj, fh, indent=4)

def save_np_compr(arr, pth):
    with open(pth, 'wb+') as fh:
        np.savez_compressed(fh, data=arr)

def load_np_compr(pth):
    return np.load(pth)["data"]

def compute_spectrum(input_image, output_shape=None):
    if output_shape == None:
        output_shape = input_image.shape
    return np.fft.fft2(input_image, s=output_shape)

def open_image(filename, resolution, allow_upsample=False):
    try:
        if filename.endswith("npz"):
            color = load_np_compr(filename)
        elif filename.endswith("webp"):
            with open(filename, 'rb') as f:
                webp_data = webp.WebPData.from_buffer(f.read())
                color = webp_data.decode(color_mode=webp.WebPColorMode.RGB)
        else:
            im = Image.open(filename)
            color = np.asarray(im).astype(float)
    except Exception as e:
        return False, None
    if len(color.shape) == 2:
        #For grayscale images
        color = np.squeeze(np.stack((color,) * 3, -1))
    elif len(color.shape) == 3 and color.shape[2] != 3:
        #For images with the wrong axis orders
        color = np.swapaxes(np.swapaxes(color, 0, 1), 1, 2)
    height, width, nc = color.shape
    #We may need to change the size
    if resolution is not None:
        width_ratio = width / resolution[1]
        height_ratio = height / resolution[0]
        if not allow_upsample and (height_ratio < 1 or width_ratio < 1):
            return False, None
        #Scale
        if height != resolution[0] or width != resolution[1]:
            if width_ratio < height_ratio:
                #We need to make the width fit first and then scale the height
                color = skimage.transform.resize(color, (int(height / width_ratio), resolution[1], color.shape[2]))
            else:
                color = skimage.transform.resize(color, (resolution[0], int(width / height_ratio), color.shape[2]))
        height, width, nc = color.shape
        #Crop
        if height != resolution[0] or width != resolution[1]:
            color = color[:resolution[0],:resolution[1],:]
    #Remove extra dimensions
    color = np.squeeze(color)
    return True, color

def get_images(logger, input_dir, min_resolution, max_images, start_image=0, ignore_resolution=False):
    filenames = []
    picture_id = 0
    all_filenames = []
    valid_endings = {"png", "npz", "jpg", "jpeg", "webp"}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_dir):
        for filename in f:
            file_extension = filename.split(".")[-1].lower()
            if file_extension in valid_endings:
                all_filenames.append(os.path.join(r, filename))
    all_filenames = sorted(all_filenames)
    logger.info("Collecting {} from {} in {}".format(max_images, len(all_filenames), input_dir))
    with tqdm(total=max_images, file=sys.stdout) as pbar:
        for filename in all_filenames:
            if picture_id >= (max_images + start_image):
                break
            if ignore_resolution:
                filenames.append(filename)
                pbar.update(1)
            else:
                valid, color = open_image(filename, (min_resolution, min_resolution))
                if not valid or color is None:
                    logger.debug(filename)
                    continue
                if picture_id >= start_image:
                    filenames.append(filename)
                    pbar.update(1)
            picture_id = picture_id + 1
    return filenames

def inc_dict_value(dictionary, value):
    if value not in dictionary:
        dictionary[value] = 0
    dictionary[value] += 1

def load_image_and_process(dataset_name, output_dir, filename, min_resolution, num_channels, evasion_level, evasion_type):
    _valid, image = open_image(filename, (min_resolution, min_resolution), allow_upsample=True)
    #Output filename for JPEG evasion
    base=os.path.splitext(os.path.basename(filename))[0]
    new_filename = os.path.join(output_dir, "{}.jpg".format(base))
    if evasion_level != 0:
        for channel_id in range(num_channels):
            image[:,:,channel_id] = evasion(image[:,:,channel_id], evasion_type, evasion_level=evasion_level, filename=new_filename)
    return image

def evasion(image, evasion_type, evasion_level=10.0, filename="image_name_2.jpg"):
    if evasion_type == "noise":
        noise_level = np.mean(np.abs(image)) * evasion_level
        return image + noise_level * np.random.randn(*image.shape)
    elif evasion_type == "blur":
        kernel = np.zeros((evasion_level, evasion_level), dtype=float)
        kernel[:,:] = 1 / (evasion_level * evasion_level)
        return scipy.signal.convolve2d(image, kernel, mode="same")
    elif evasion_type == "jpeg":
        imageio.imwrite(filename, image, quality=evasion_level)
        return_value = imageio.imread(filename)
        os.remove(filename)
        return return_value
    return image

def complex_correlation(freq_a, freq_b):
    return np.mean(np.nan_to_num(np.divide(np.multiply(freq_a, np.conj(freq_b)).real, np.multiply(np.abs(freq_a), np.abs(freq_b)))))

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def sort_and_count(positive_results, negative_results):
    dict_positive_results = {}
    for result in positive_results:
        inc_dict_value(dict_positive_results, result)
    dict_negative_results = {}
    for result in negative_results:
        inc_dict_value(dict_negative_results, result)
    x_values = np.array(sorted(list(set(positive_results + negative_results))))
    positive_values = np.zeros(x_values.shape, dtype=int)
    negative_values = np.zeros(x_values.shape, dtype=int)
    for i in range(x_values.size):
        threshold = x_values[i]
        if threshold in dict_positive_results:
            positive_values[i] = dict_positive_results[threshold]
        if threshold in dict_negative_results:
            negative_values[i] = dict_negative_results[threshold]
    return x_values, positive_values, negative_values

def compute_statistics(logger, positives, negatives):
    x_values, positive_results, negative_results = sort_and_count(positives, negatives)
    tpr_results = []
    fpr_results = []
    precision_results = []
    accuracy_results = []
    total_true = np.sum(positive_results)
    total_fake = np.sum(negative_results)
    for i in range(x_values.size):
        tp = np.sum(positive_results[i:])
        fn = np.sum(positive_results[:i])
        tn = np.sum(negative_results[:i])
        fp = np.sum(negative_results[i:])
        tpr = tp / (total_true) * 100
        fpr = fp / (total_fake) * 100
        precision = tp / (tp + fp) * 100
        accuracy = (tp + tn) / (total_fake + total_true) * 100
        fpr_results.append(fpr)
        tpr_results.append(tpr)
        precision_results.append(precision)
        accuracy_results.append(accuracy)
    return x_values, np.array(tpr_results), np.array(fpr_results), np.array(precision_results), np.array(accuracy_results)

def load_datasets_info():
    all_datasets = load_json("datasets.json")
    datasets_map = {}
    real_datasets = set()
    for dataset in all_datasets:
        datasets_map[dataset["dataset_name"]] = dataset
        if dataset["type"] == "real":
            real_datasets.add(dataset["dataset_name"])
    return datasets_map, datasets_map.keys(), real_datasets
