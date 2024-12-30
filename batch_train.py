import os
import time
import sys
import pickle
import torch.multiprocessing as mp
import torch
import logging
module_logger = logging.getLogger(__name__)
import cv2
import argparse
import yaml
import pprint
from itertools import product
import torch.nn as nn

import main
import constants
from training import datasets
import utils

def base_config(base_name, mincam_size, model_type, device, img_size,
                label_select, hidden_layer_size,
                hidden_layer_count, lr, 
                mask_init_method, epochs,
                regression_loss, simulate_pd_area_blur,
                mask_blur_kernel_sigma, simulate_directivity,
                mincam_sensor_gain,
                mincam_sensor_saturation_val,
                mincam_sensor_n_bits,
                mincam_read_noise_std,
                mask_min_value,
                mask_max_value,
                model_vert_fov,
                model_horiz_fov,
                dataset_name,
                early_stop_epochs,
                checkpoint_every_n_epochs,
                checkpoint_minibatches_until,
                train_augmentations,
                batch_size):

    # Create configuration dictionaries
    dataset_options = {
        "gpu_dataset": True,
        "batch_size": batch_size,
        "num_workers": 0,
        "shuffle": True,
        "label_select": label_select,
        "img_size": img_size,
        "dataset_name": dataset_name
    }

    # Set regression_metrics
    if label_select == "people_count":
        regression_metrics = True
    else:
        regression_metrics = False
    dataset_options["regression_metrics"] = regression_metrics

    model_options = {
        "mincam_size": mincam_size,
        "base_exp_name": base_name,
        "model_type": model_type,
        "hidden_layer_size": hidden_layer_size,
        "hidden_layer_count": hidden_layer_count,
        "mincam_sensor_gain": mincam_sensor_gain,
        "mincam_sensor_saturation_val": mincam_sensor_saturation_val,
        "mincam_sensor_n_bits": mincam_sensor_n_bits,
        "mincam_realistic_sensor": True,
        "mincam_read_noise_std": mincam_read_noise_std,
        "mask_init_method": mask_init_method,
        "simulate_pd_area_blur": simulate_pd_area_blur,
        "mask_blur_kernel_sigma": mask_blur_kernel_sigma,
        "simulate_directivity": simulate_directivity,
        "mask_min_value": mask_min_value,
        "mask_max_value": mask_max_value,
        "model_vert_fov": model_vert_fov,
        "model_horiz_fov": model_horiz_fov,
    }

    if model_type == "mincam" or model_type == "mincam_binary" or \
        model_type == "mincam_fixed_mask" or model_type == "mincam_refine":
        model_options["mincam_size"] = mincam_size

    train_options = {
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "resume_from_checkpoint": False,
        "quiet": True,
        "lr_scheduler": False,
        "warmup_lr": False,
        "augmentation_fn": datasets.DataAugmentation() if train_augmentations else nn.Identity(),
        "regression_loss": regression_loss,
        "early_stop_epochs": early_stop_epochs,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
        "checkpoint_minibatches_until": checkpoint_minibatches_until
    }

    numeric_log_level = getattr(logging, "WARN", None)
    logging.basicConfig(level=numeric_log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)])

    return dataset_options, model_options, train_options


def run_config(dataset_options, model_options, train_options):
    """
    Train the model defined by the given options.
    """
    main.init_libraries()

    # Load data
    train_dataloader, val_dataloader = \
        main.load_training_data(model_options, train_options, dataset_options)

    main.run(model_options, train_options, dataset_options, train_dataloader,
             val_dataloader)

def run_jobs(jobs):
    for job in jobs:
        run_config(*job)

def save_results(base_exp_name, model_type, model_names, mincam_sizes,
                 test_results, mask_pyramid):
    results_parent = os.path.join(constants.RESULTS_PATH, "%s-%s" %
                                  (base_exp_name, model_type))
    os.makedirs(results_parent, exist_ok=True)

    module_logger.info("Saving group results to:\n%s" % results_parent)

    # save results for plots in a pickle file
    with open(os.path.join(results_parent, "results.pkl"), "wb") as f:
        D = {
            "model_names": model_names,
            "mincam_sizes": mincam_sizes,
            "test_results": test_results,
        }
        pickle.dump(D, f)

    # save the pyramid of masks in a subfolder
    if model_type != "baseline":
        for masks in mask_pyramid:
            N = masks.shape[0]
            mask_dir = os.path.join(results_parent, "mask-pyramid", str(N))
            os.makedirs(mask_dir, exist_ok=True)
            for i in range(N):
                cv2.imwrite(os.path.join(mask_dir, "%d.png" % i), masks[i,:,:])


def create_jobs_from_config(group_experiment_config):
    """
    Train a set of mincams.
    """
    ### Experiment configuration
    model_type = group_experiment_config["model_type"]
    epochs = group_experiment_config["epochs"]
    label_select = group_experiment_config["label_select"]
    hidden_layer_sizes = group_experiment_config["hidden_layer_sizes"]
    hidden_layer_counts = group_experiment_config["hidden_layer_counts"]
    lrs = group_experiment_config["lrs"]
    exp_prefix = group_experiment_config["exp_prefix"]
    cam_sizes = group_experiment_config["cam_sizes"]
    img_sizes = group_experiment_config["img_sizes"]
    mask_init_methods = group_experiment_config["mask_init_method"]
    regression_loss = group_experiment_config["regression_loss"]
    simulate_pd_area_blur = group_experiment_config["simulate_pd_area_blur"]
    mask_blur_kernel_sigma = group_experiment_config["mask_blur_kernel_sigma"]
    simulate_directivity = group_experiment_config["simulate_directivity"]
    mincam_sensor_gain = group_experiment_config["mincam_sensor_gain"]
    mincam_sensor_saturation_val = group_experiment_config["mincam_sensor_saturation_val"]
    mincam_sensor_n_bits = group_experiment_config["mincam_sensor_n_bits"]
    mincam_read_noise_std = group_experiment_config["mincam_read_noise_std"]
    mask_min_value = group_experiment_config["mask_min_value"]
    mask_max_value = group_experiment_config["mask_max_value"]
    model_vert_fov = group_experiment_config["model_vert_fov"]
    model_horiz_fov = group_experiment_config["model_horiz_fov"]
    dataset_name = group_experiment_config["dataset_name"]
    early_stop_epochs = group_experiment_config["early_stop_epochs"]
    checkpoint_every_n_epochs = group_experiment_config["checkpoint_every_n_epochs"]
    checkpoint_minibatches_until = group_experiment_config["checkpoint_minibatches_until"]
    train_augmentations = group_experiment_config["train_augmentations"]
    batch_size_list = group_experiment_config["batch_size"]

    jobs_per_gpu = group_experiment_config["jobs_per_gpu"]

    ### Create jobs
    jobs = [[] for _ in range(torch.cuda.device_count() * jobs_per_gpu)]
    d = 0
    job_i = 0

    for (cam_size, img_size), hidden_layer_count, hidden_layer_size, \
        mask_init_method, \
        blur_kernel, lr, batch_size in \
        product(zip(cam_sizes, img_sizes), hidden_layer_counts,
                hidden_layer_sizes,
                mask_init_methods,
                mask_blur_kernel_sigma, 
                lrs, 
                batch_size_list):

        base_exp_name = exp_prefix
        if model_type != "conv":
            base_exp_name += "-%dx%d" % (hidden_layer_size, hidden_layer_count)

        if len(mask_init_methods) > 1:
            base_exp_name += "-%s" % mask_init_method

        if len(batch_size_list) > 1:
            base_exp_name += "-b%d" % batch_size

        if "mincam" in model_type:
            base_exp_name += "-mblur%d" % blur_kernel

        if simulate_directivity:
            base_exp_name += "-directivity"

        base_exp_name += "-lr-{:.0e}".format(lr)

        device = torch.device("cuda:%s" % d)
        dataset_options, model_options, train_options = base_config(\
                base_exp_name, cam_size, model_type,
                device, img_size, label_select,
                hidden_layer_size, hidden_layer_count,
                lr, 
                mask_init_method, epochs,
                regression_loss, simulate_pd_area_blur, blur_kernel,
                simulate_directivity,
                mincam_sensor_gain,
                mincam_sensor_saturation_val,
                mincam_sensor_n_bits,
                mincam_read_noise_std,
                mask_min_value,
                mask_max_value,
                model_vert_fov,
                model_horiz_fov,
                dataset_name,
                early_stop_epochs,
                checkpoint_every_n_epochs,
                checkpoint_minibatches_until,
                train_augmentations,
                batch_size)

        job = (dataset_options, model_options, train_options)
        jobs[job_i].append(job)

        d = (d + 1) % torch.cuda.device_count()
        job_i = (job_i + 1) % len(jobs)

    return jobs

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    parser.add_argument("-f", "--config_file", type=str,
                        help="Experiment configuration file (.yml)")
    return parser.parse_args()

def load_experiment_configuration():
    args = parse_args()

    fname = constants.EXP_CONFIGS_PATH / args.config_file
    with open(fname, "r") as f:
        D = yaml.safe_load(f)

    group_experiment_configs = []
    for exp_id in D.keys():
        if exp_id == "global_options":
            global_options = utils.list_of_dicts_to_dict(D[exp_id])
        else:
            exp_config = D[exp_id]
            config_dict = {**utils.list_of_dicts_to_dict(exp_config),
                           **global_options}
            # Convert image and camera sizes to a list of tuple
            if config_dict["model_type"] == "mincam":
                config_dict["cam_sizes"] = \
                    [s for s in config_dict["cam_sizes"]]
            else:
                config_dict["cam_sizes"] = \
                    [tuple(s) for s in config_dict["cam_sizes"]]
            config_dict["img_sizes"] = \
                [tuple(s) for s in config_dict["img_sizes"]]
            group_experiment_configs.append(config_dict)

    module_logger.info("Experiment Configurations")
    pp = pprint.PrettyPrinter(indent=4)
    module_logger.info("\n" + pp.pformat(group_experiment_configs))
    module_logger.info("")

    return group_experiment_configs

def run_batch_train():
    mp.set_start_method("spawn")
    group_experiment_configs = load_experiment_configuration()

    ### Create jobs
    jobs_per_gpu = None
    for c in group_experiment_configs:
        if jobs_per_gpu is None:
            jobs_per_gpu = create_jobs_from_config(c)
        else:
            jobs_new = create_jobs_from_config(c)
            for i in range(len(jobs_new)):
                jobs_per_gpu[i].extend(jobs_new[i])
    
    ### Start jobs
    DEBUG = True
    if DEBUG:
        for job_list in jobs_per_gpu:
            run_jobs(job_list)
    else:
        delay_between_spawn = 0 # seconds
        processes = []
        for job_list in jobs_per_gpu:
            p = mp.Process(target=run_jobs, args=(job_list,))
            p.start()
            processes.append(p)
            time.sleep(delay_between_spawn)

        for p in processes:
            p.join()


if __name__ == "__main__":
    run_batch_train()
