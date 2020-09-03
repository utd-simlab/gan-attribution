#!/usr/bin/env python3

import logging
import os

import random

from . import utils

def process_split(args):
    #Create logger
    utils.make_dir(args.output_root)
    logger = utils.open_logger(os.path.join(args.output_root, "log"), logging.INFO)

    fingerprint_samples = args.fingerprint
    testing_samples = args.testing
    allow_upsample = args.allow_upsample
    overwrite = args.overwrite

    #Make the output dirs
    split_dir = os.path.join(args.output_root, args.split)
    utils.make_dir(split_dir)

    datasets_map, datasets_list, real_datasets = utils.load_datasets_info()

    for dataset_name in datasets_map:
        logger.info("Computing split for {}".format(dataset_name))
        total_images = testing_samples
        if datasets_map[dataset_name]["fingerprint"]:
            total_images += fingerprint_samples
        filename_list_path = os.path.join(split_dir, "{}_filenames.json".format(dataset_name))
        if not overwrite and os.path.exists(filename_list_path):
            continue
        filenames = utils.get_images(logger, datasets_map[dataset_name]["directory"], datasets_map[dataset_name]["resolution"], total_images, start_image=0, ignore_resolution=allow_upsample)
        if len(filenames) != total_images:
            raise Exception("Not enough images for {}".format(dataset_name))
        random.shuffle(filenames)
        split = {
                    "testing": filenames[:testing_samples],
                    "fingerprint": filenames[testing_samples:testing_samples+fingerprint_samples]}
        utils.save_json(split, filename_list_path)
