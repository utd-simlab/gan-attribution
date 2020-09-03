#!/usr/bin/env python3

import logging
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

from . import utils

def process_fingerprint(args):
    #Create logger
    utils.make_dir(args.output_root)
    logger = utils.open_logger(os.path.join(args.output_root, "log"), logging.INFO)

    #Make the output dirs
    fingerprint_dir = os.path.join(args.output_root, args.fingerprint)
    split_dir = os.path.join(args.output_root, args.split)
    utils.make_dir(fingerprint_dir)
    overwrite = args.overwrite

    num_channels = 3 #Grayscale is not implemented, but this is parameterized to allow introducing grayscale in open_image

    datasets_map, datasets_list, real_datasets = utils.load_datasets_info()

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        #Compute the fingerprint
        for dataset_name in datasets_map:
            if datasets_map[dataset_name]["fingerprint"]:
                logger.info("Loading split for {}".format(dataset_name))
                filename_list_path = os.path.join(split_dir, "{}_filenames.json".format(dataset_name))
                if not os.path.exists(filename_list_path):
                    raise Exception("Unable to find split file for {} {}".format(dataset_name, filename_list_path))
                split = utils.load_json(filename_list_path)
                fingerprint_filename = os.path.join(fingerprint_dir, "{}_fingerprint.npz".format(dataset_name))
                if not overwrite and os.path.exists(fingerprint_filename):
                    continue
                compute_fingerprint(logger, pool, num_processes, dataset_name, num_channels, fingerprint_filename, split["fingerprint"], datasets_map[dataset_name]["resolution"])

def partial_fingerprint(filenames, num_channels, min_resolution):
    fingerprint = np.zeros((num_channels, min_resolution, min_resolution), dtype=np.complex128)
    for filename in tqdm(filenames):
        _valid, image = utils.open_image(filename, (min_resolution, min_resolution), allow_upsample=True)
        #Compute the spectrums
        for channel_id in range(num_channels):
            spectrum = utils.compute_spectrum(image[:,:,channel_id])
            fingerprint[channel_id,:,:] = fingerprint[channel_id,:,:] + spectrum / len(filenames)
    return fingerprint

def compute_fingerprint(logger, pool, num_processes, dataset_name, num_channels, fingerprint_filename, filenames, min_resolution):
    logger.info("compute_fingerprint: {}".format(dataset_name))
    #Get the filenames
    filename_parts = utils.split(filenames, num_processes)

    #Build the args list for computing the partial fingerprint
    args = []
    for part in filename_parts:
        args.append((part, num_channels, min_resolution))
    partial_fingerprint_results = pool.starmap(partial_fingerprint, args)

    #Combine the partial fingerprints
    fingerprint = np.zeros((num_channels, min_resolution, min_resolution), dtype=np.complex128)
    for channel_id in range(num_channels):
        for partial_fingerprint_result in partial_fingerprint_results:
            fingerprint[channel_id,:,:] = fingerprint[channel_id,:,:] + partial_fingerprint_result[channel_id,:,:] / num_processes

    utils.save_np_compr(fingerprint, fingerprint_filename)
