#!/usr/bin/env python3

import logging
import multiprocessing
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import utils

def process_testing(args):
    #Create logger
    utils.make_dir(args.output_root)
    logger = utils.open_logger(os.path.join(args.output_root, "log"), logging.INFO)

    #Make the output dirs
    split_dir = os.path.join(args.output_root, args.split)
    fingerprint_dir = os.path.join(args.output_root, args.fingerprint)
    correlations_dir = os.path.join(args.output_root, args.correlations)
    prediction_dir  = os.path.join(args.output_root, args.predictions)
    temp_output_dir = os.path.join(args.output_root, args.temp)
    utils.make_dir(correlations_dir)
    utils.make_dir(temp_output_dir)
    utils.make_dir(prediction_dir)
    overwrite = args.overwrite

    #Correlation options
    threshold = args.threshold
    evasion_type = args.evasion_type
    if evasion_type == "noise": #Noise levels
        evasion_level = float(args.evasion_level)
        if evasion_level < 0 or evasion_level > 1:
            raise Exception("Bad noise level. Must be between 0 and 1.")
    elif evasion_type == "blur": #Blur levels
        evasion_level = int(args.evasion_level)
        if evasion_level <= 0:
            raise Exception("Bad blur level. Must be greater than 0.")
    elif evasion_type == "jpeg": #JPEG levels
        evasion_level = int(args.evasion_level)
        if evasion_level < 0 or evasion_level > 100:
            raise Exception("Bad jpeg level. Must be between 1 and 100.")
    else: #No evasion
        evasion_level = 0

    #Intermediate results
    num_channels = 3 #Grayscale is not implemented, but this is parameterized to allow introducing grayscale in open_image

    datasets_map, datasets_list, real_datasets = utils.load_datasets_info()

    pairs = []
    if args.test_cases in ["both", "one_to_one"]:
        #One to one pairs
        for dataset_name in datasets_map:
            if dataset_name in real_datasets:
                continue
            pairs.append({  "correlation_name": "{}_one_to_one".format(dataset_name), "fingerprint": dataset_name,
                            "datasets": [datasets_map[dataset_name]["training_dataset"], dataset_name],
                            "min_resolution": datasets_map[dataset_name]["resolution"]})
    if args.test_cases in ["both", "all_to_all"]:
        #All to all pairs
        for dataset_name in datasets_map:
            if dataset_name in real_datasets:
                continue
            pairs.append({  "correlation_name": "{}_all_to_all".format(dataset_name), "fingerprint": dataset_name,
                            "datasets": datasets_list, "min_resolution": datasets_map[dataset_name]["resolution"]})

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        #Perform the correlation checking
        for pair in pairs:
            #Check if results have been computed already
            correlations = {}
            correlations_filename = os.path.join(correlations_dir, "{}.json".format(pair["correlation_name"]))
            if not overwrite and os.path.exists(correlations_filename):
                continue
            #Load fingerprint
            fingerprint_filename = os.path.join(fingerprint_dir, "{}_fingerprint.npz".format(pair["fingerprint"]))
            if not os.path.exists(fingerprint_filename):
                raise Exception("Unable to find fingerprint {}".format(fingerprint_filename))
            fingerprint = utils.load_np_compr(fingerprint_filename)
            datasets_filenames_list = {}
            for dataset_name in pair["datasets"]:
                filename_list_path = os.path.join(split_dir, "{}_filenames.json".format(dataset_name))
                if not os.path.exists(filename_list_path):
                    raise Exception("Unable to find split file for {} {}".format(dataset_name, filename_list_path))
                split = utils.load_json(filename_list_path)
                datasets_filenames_list[dataset_name] = split["testing"]
            correlations = correlation_with_fingerprint(logger, pool, num_processes, fingerprint, num_channels,
                                                        evasion_level, evasion_type, correlations, pair["fingerprint"], correlations_filename,
                                                        datasets_filenames_list, real_datasets, pair["min_resolution"], temp_output_dir, pair)
            utils.save_json(correlations, correlations_filename)
        for pair in pairs:
            correlations_filename = os.path.join(correlations_dir, "{}.json".format(pair["correlation_name"]))
            if not os.path.exists(correlations_filename):
                continue
            correlations = utils.load_json(correlations_filename)
            output_predictions(logger, prediction_dir, threshold, correlations, pair["fingerprint"],
                                pair["datasets"], pair["correlation_name"], evasion_level, evasion_type)

def partial_correlation_with_fingerprint(filenames, dataset_name, output_dir, min_resolution,
                                        evasion_level, evasion_type, num_channels, fingerprint):
    results = []
    for filename in tqdm(filenames):
        #Apply the evasion
        image = utils.load_image_and_process(dataset_name, output_dir, filename, min_resolution, num_channels, evasion_level, evasion_type)
        image_correlation_avg = 0
        for channel_id in range(num_channels):
            #Compute the spectrums
            freq = utils.compute_spectrum(image[:,:,channel_id])
            result = utils.complex_correlation(fingerprint[channel_id,:,:], freq)
            #Plot the average correlation matrix
            image_correlation_avg = image_correlation_avg + result / num_channels
        results.append({"dataset": dataset_name, "filename": filename, "correlation": image_correlation_avg})
    return results

def correlation_with_fingerprint(   logger, pool, num_processes, fingerprint, num_channels, evasion_level, evasion_type,
                                correlations, fingerprint_name, correlations_filename,
                                datasets_filenames_list, real_datasets, min_resolution, output_dir, pair):
    logger.info("correlation_with_fingerprint: {}".format(pair))
    for dataset_name in datasets_filenames_list:
        logger.info("Computing correlation for {} {} {}".format(fingerprint_name, dataset_name, evasion_level))
        results_key = "{}_{}_{}".format(fingerprint_name, dataset_name, evasion_level)
        if results_key in correlations:
            continue
        #Get the filename list
        filenames = datasets_filenames_list[dataset_name]
        filename_parts = utils.split(filenames, num_processes)

        #Build the args list for computing the partial average
        args = []
        for part in filename_parts:
            if dataset_name in real_datasets: #We do not use any evasion for real images
                args.append((part, dataset_name, output_dir, min_resolution,
                            0, "noise", num_channels, fingerprint))
            else:
                args.append((part, dataset_name, output_dir, min_resolution,
                            evasion_level, evasion_type, num_channels, fingerprint))
        partial_correlations_results = pool.starmap(partial_correlation_with_fingerprint, args)

        #Combine the partial results
        results_list = []
        for partial_correlations_result in partial_correlations_results:
            results_list.extend(partial_correlations_result)

        #Save the results to the dictionary
        correlations[results_key] = results_list
        utils.save_json(correlations, correlations_filename)
    return correlations

def output_predictions(logger, output_dir, threshold, correlations, fingerprint_name, datasets_list, correlation_name, evasion_level, evasion_type):
    all_results = []
    fingerprint_key = "{}_{}_{}".format(fingerprint_name, fingerprint_name, evasion_level)
    if threshold is None:
        all_opt_thresholds = []
        for dataset_name in datasets_list:
            if dataset_name == fingerprint_name:
                continue
            dataset_key = "{}_{}_{}".format(fingerprint_name, dataset_name, evasion_level)
            fingerprint_results = [result["correlation"] for result in correlations[fingerprint_key]]
            other_results       = [result["correlation"] for result in correlations[dataset_key]]
            thresholds, tpr_results, fpr_results, precision_results, accuracy_results = utils.compute_statistics(logger, fingerprint_results, other_results)
            opt_threshold_idx = np.max(np.where(accuracy_results == np.amax(accuracy_results)))
            all_opt_thresholds.append(thresholds[opt_threshold_idx])
        threshold = np.mean(all_opt_thresholds)
    logger.info("Output predictions for {}".format(correlation_name))
    correct_count = 0
    total_count = 0
    fp_count = 0
    negative_count = 0
    for dataset_name in datasets_list:
        dataset_key = "{}_{}_{}".format(fingerprint_name, dataset_name, evasion_level)
        for result in correlations[dataset_key]:
            all_results.append(result)
            if result["correlation"] < threshold:
                all_results[-1]["predicted_from_model"] = False
            else:
                all_results[-1]["predicted_from_model"] = True
            if dataset_name == fingerprint_name:
                all_results[-1]["is_from_model"] = True
            else:
                all_results[-1]["is_from_model"] = False
            all_results[-1]["threshold"] = threshold
            all_results[-1]["fingerprint"] = fingerprint_name
            if all_results[-1]["predicted_from_model"] == all_results[-1]["is_from_model"]:
                correct_count += 1
            total_count += 1
            if dataset_name != fingerprint_name:
                negative_count += 1
                if all_results[-1]["predicted_from_model"] != all_results[-1]["is_from_model"]:
                    fp_count += 1
    output_filename = os.path.join(output_dir, "predictions_{}.json".format(correlation_name))
    utils.save_json(all_results, output_filename)
    logger.info("Accuracy of {} for {}".format(correct_count/total_count * 100, correlation_name))
    # logger.info("False Positive Rate of {} for {}".format(fp_count/negative_count * 100, correlation_name))
