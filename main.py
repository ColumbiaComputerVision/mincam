import sys
import os
import gc
import random
import argparse
import datetime
import time
import pickle
import numpy as np
import torchvision.utils
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cv2
import sklearn.metrics
import logging
module_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
from pathlib import Path
import pprint
import shutil
from typing import Dict

from constants import LOG_PATH, MODEL_PATH
from training import datasets, camera_models, loss, early_stop
import utils


TB_ADD_MASKS = True
TB_ADD_INITIAL_MASKS = True
TB_ADD_MASKS_EVERY_N = 100
PRINT_STATISTICS_EVERY_N = 1


def forward_pass(model: camera_models.Baseline | camera_models.Mincam,
                 dataloader: Dict,
                 train_options: Dict,
                 dataset_options: Dict,
                 loss_fn: loss.Loss,
                 train=True,
                 writer=None,
                 optimizer=None,
                 scheduler=None,
                 val_loader=None,
                 minibatch_i=0):
    device = train_options["device"]

    regression_loss = train_options["regression_loss"]

    if train:
        assert optimizer is not None

        if "augmentation_fn" in train_options.keys():
            augmentation_fn = train_options["augmentation_fn"]
        else:
            augmentation_fn = None

    if dataloader.drop_last:
        dataset_size = len(dataloader) * dataloader.batch_size
    else:
        dataset_size = len(dataloader.dataset)

    # Raw classifier output for each sample
    if regression_loss:
        labels_pred = torch.zeros(dataset_size, device=device)
        logits_pred = None
    else:
        logits_pred = torch.zeros((dataset_size, dataset_options["num_classes"]),
                                  device=device)
    # Ground truth label
    labels_gt = torch.zeros(dataset_size, device=device, dtype=torch.int64)
    # Total epoch loss
    loss_epoch = {
        # Mean loss over the epoch
        "loss_epoch": torch.tensor(0.0, device=device),
        "loss_task": torch.tensor(0.0, device=device),
    }

    i = 0
    for sample in dataloader:
        # Load minibatch
        x, y = sample
        x = x.to(device)
        y = y.to(device)

        if train and augmentation_fn is not None:
            x = augmentation_fn(x)

        # Forward pass
        y_pred = model(x)
        if regression_loss:
            y_pred = y_pred.squeeze()
            y = y.to(torch.float32)

        # Loss function
        loss_dict = loss_fn(model, y_pred, y)

        # Backward pass and optimizer update
        if train:
            optimizer.zero_grad()
            loss_dict["loss_batch"].backward()
            optimizer.step()

        # Save predictions
        N = x.shape[0]

        if y.ndim > 1: # ground-truth labels may be a PMF
            y = y.argmax(1)
        labels_gt[i:(i+N)] = y
        if regression_loss:
            labels_pred[i:(i+N)] = y_pred
        else:
            logits_pred[i:(i+N)] = y_pred

        # Update epoch loss
        loss_epoch["loss_epoch"] += loss_dict["loss_batch"].detach() * N
        loss_epoch["loss_task"] += loss_dict["loss_task"].detach() * N

        # If training, update iter loss
        if train and train_options["checkpoint_minibatches_until"] > 0:
            if minibatch_i % 100 == 0:
                update_iter_loss(model, val_loader, train_options,
                                 dataset_options, loss_fn, optimizer, scheduler,
                                 writer, loss_dict, minibatch_i)
            minibatch_i += 1

        i += N


    if not regression_loss:
        labels_pred = logits_pred.argmax(1)
    assert i == dataset_size

    # Epoch steps
    if train and scheduler is not None:
        scheduler.step()

    # Divide all loss components by number of batches and transfer to CPU
    for k in loss_epoch.keys():
        loss_epoch[k] = loss_epoch[k].cpu() / dataset_size

    return loss_epoch, labels_pred.cpu(), labels_gt.cpu(), logits_pred, \
        minibatch_i

@torch.no_grad()
def update_iter_loss(model, val_loader, train_options, dataset_options,
                     loss_fn, optimizer, scheduler, writer, loss_dict,
                     minibatch_i):
    # Compute validation loss
    was_training = model.training
    with torch.no_grad():
        model.eval()
        val_loss_dict, val_labels_pred, val_labels_gt, _, _ = forward_pass(
            model, val_loader, train_options, dataset_options, loss_fn,
            train=False)
    if was_training:
        model.train()

    writer.add_scalar("Loss/iter_train", loss_dict["loss_batch"], minibatch_i)
    writer.add_scalar("Loss/iter_val", val_loss_dict["loss_epoch"], minibatch_i)

    # Compute accuracy / rmse
    if dataset_options["regression_metrics"]:
        val_rmse = rmse(val_labels_pred, val_labels_gt)
        writer.add_scalar("RMSE/iter_val", val_rmse, minibatch_i)
    else:
        val_acc = sklearn.metrics.accuracy_score(
            val_labels_gt.cpu().numpy(), val_labels_pred.cpu().numpy())
        val_bal_acc = sklearn.metrics.balanced_accuracy_score(
            val_labels_gt.cpu().numpy(), val_labels_pred.cpu().numpy())
        writer.add_scalar("Accuracy/iter_val", val_acc, minibatch_i)
        writer.add_scalar("Balanced Accuracy/iter_val", val_bal_acc, minibatch_i)

    # save minibatch checkpoint
    if minibatch_i < train_options["checkpoint_minibatches_until"]:
        save_minibatch_checkpoint(minibatch_i, model, optimizer, scheduler,
                                  loss_fn, train_options["exp_name"])

@torch.no_grad()
def forward_pass_on_dataset(model, D, train_options):
    """
    Forward pass on the given dataset for inference. Do not use for training.
    """
    model.eval()

    device = train_options["device"]
    N = len(D)
    y_pred = torch.zeros(N, device=device)
    y_gt = torch.zeros_like(y_pred)

    for i in range(N):
        img, label = D[i]
        img = img.to(device)
        label = label.to(device)

        img = img[None,None,:,:]
        label_pred = model(img)
        label_pred = label_pred.argmax()

        y_pred[i] = label_pred

        if label.ndim > 1:
            label = label.argmax(1)
        y_gt[i] = label

    return y_pred, y_gt

@torch.no_grad()
def print_validation_performance(model, val_loader, train_options,
                                 dataset_options, loss_fn):
    model.eval()
    val_loss_epoch, val_labels_pred, val_labels_gt, _, _ = forward_pass(
        model, val_loader, train_options, dataset_options, loss_fn, train=False)

    module_logger.info("Initial Model")
    module_logger.info("  Val loss       : %4f" % val_loss_epoch["loss_epoch"])

    if dataset_options["regression_metrics"]:
        module_logger.info("  Val RMSE       : %4f" %
                           rmse(val_labels_pred, val_labels_gt))
    else:
        val_acc = sklearn.metrics.accuracy_score(
            val_labels_gt.cpu().numpy(), val_labels_pred.cpu().numpy())
        val_bal_acc = sklearn.metrics.balanced_accuracy_score(
            val_labels_gt.cpu().numpy(), val_labels_pred.cpu().numpy())
        module_logger.info("  Val acc        : %.2f%%" % (val_acc * 100))
        module_logger.info("  Val bal acc    : %.2f%%\n" % (val_bal_acc * 100))

def rmse(l_pred, l_gt):
    l_pred = l_pred.to(torch.float32)
    l_gt = l_gt.to(torch.float32)
    return torch.sqrt(((l_pred - l_gt)**2).mean())

def tensorboard_visualize_mask(model, writer, epoch, ex_img, initial_mask=False):
    """
    Send gamma-corrected images to Tensorboard for visualization
    """
    mask_imgs = model.visualize_mask(-1) # Nx3xHxW

    prefix = "Initial " if initial_mask else ""
    if mask_imgs is not None:
        num_masks = mask_imgs.shape[0]
        nrows = int(np.sqrt(num_masks))
        mask_imgs_grid = torchvision.utils.make_grid(
            mask_imgs**(1/2.2), value_range=(0, 1),
            nrow=nrows).cpu()
        writer.add_image(prefix + "Mask", mask_imgs_grid, global_step=epoch,
                         dataformats="CHW")

    if mask_imgs is not None:
        # Masked images
        masked_imgs = mask_imgs.clone()
        masked_imgs[:,2,:,:] = ex_img[None,:,:]

        masked_imgs = torchvision.utils.make_grid(
            masked_imgs**(1/2.2), value_range=(0, 1),
            nrow=nrows).cpu()

        writer.add_image(prefix + "Masked Image", masked_imgs,
                         global_step=epoch, dataformats="CHW")

def save_mask_imgs(model, parent_path: Path, mask_mask=None):
    if not parent_path.is_dir():
        parent_path.mkdir()

    mask_imgs = model.visualize_mask(-1)
    if mask_imgs is None:
        # return if the camera does not support mask images
        return

    mask_imgs = np.transpose(mask_imgs.cpu().numpy(), (0, 2, 3, 1))

    if mask_mask is not None:
        mask_imgs = mask_imgs * mask_mask[None,:,:,None]

    N = mask_imgs.shape[0]
    for i in range(N):
        # Gamma correct
        img = (mask_imgs[i]**(1/2.2))
        img = (img * 255).astype(np.uint8)

        # Upsample with nearest-neighbor interpolation
        img = utils.upsample_nn(img)

        # Save
        cv2.imwrite(str(parent_path / ("%d.png" % i)), img)


def _most_recent_checkpoint(exp_name):
    """
    Get the epoch of the recent checkpoint from the model's checkpoint
    directory
    """
    checkpoint_files = MODEL_PATH / exp_name / "checkpoints"
    chkpts = []
    for f in checkpoint_files.iterdir():
        if f.suffix == ".pt" and f.stem.isdigit():
            chkpts.append(int(f.stem))
    return max(chkpts)

def init_training_state(resume_from_checkpoint, model, optimizer, scheduler,
                        loss_fn, exp_name, train_device):
    """
    Loads model and optimizer state from a training checkpoint.
    If resume_from_checkpoint is not set, returns initial values.
    """
    if resume_from_checkpoint:
        D_checkpoint = torch.load(
            MODEL_PATH / exp_name / "checkpoints" /
            ("%d.pt" % _most_recent_checkpoint(exp_name)),
            map_location=train_device)
        epoch_start = D_checkpoint['epoch'] + 1
        model.load_state_dict(D_checkpoint['model'])
        optimizer.load_state_dict(D_checkpoint['optimizer'])
        loss_fn.load_state_dict(D_checkpoint['loss'])
        if scheduler is not None:
            scheduler.load_state_dict(D_checkpoint['scheduler'])
        module_logger.info("Resuming training at epoch %d" % epoch_start)
    else:
        epoch_start = 0

    return epoch_start

def write_ex_img_to_tensorboard(train_loader, writer, train_options):
    imgs = [train_loader.dataset[i][0] for i in
            np.random.choice(np.arange(len(train_loader.dataset)), size=36,
                             replace=False)]
    imgs = torch.stack(imgs)
    if imgs.ndim == 3:
        imgs = imgs[:,None,:,:]

    imgs_aug = train_options["augmentation_fn"](imgs)

    imgs = torch.clamp(imgs, 0, 1)
    imgs = imgs**(1/2.2)
    imgs = torchvision.utils.make_grid(imgs, value_range=(0, 1), nrow=6)

    imgs_aug = torch.clamp(imgs_aug, 0, 1)
    imgs_aug = imgs_aug**(1/2.2)
    imgs_aug = torchvision.utils.make_grid(imgs_aug, value_range=(0, 1), nrow=6)

    writer.add_image("Training Images", imgs, dataformats="CHW")
    writer.add_image("Training Images (Augmented)", imgs_aug, dataformats="CHW")

def create_optimizer(model, train_options):
    lr = train_options["lr"]

    if isinstance(model, camera_models.Mincam):
        optimizer = torch.optim.Adam(model.parameters(),
            lr=lr, weight_decay=0, betas=(0.9, 0.999), foreach=True)
    elif isinstance(model, camera_models.Baseline):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999),
            foreach=True)
    else:
        module_logger.error("Unsupported model type in create_optimizer()")
        sys.exit(1)

    return optimizer

def create_lr_schedulers(optimizer, train_options):
    schedulers = []

    if train_options["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [150], gamma=0.2
        )
        schedulers.append(scheduler)
    if train_options["warmup_lr"]:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1, total_iters=100
        )
        schedulers.append(warmup_scheduler)

    if len(schedulers) > 0:
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)
        return scheduler
    else:
        return None

def save_epoch_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                          exp_name):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None
            else None,
        'loss': loss_fn.state_dict()
    },
    MODEL_PATH / exp_name / "checkpoints" / ("%d.pt" % epoch))

def save_minibatch_checkpoint(minibatch_i, model, optimizer, scheduler, loss_fn,
                              exp_name):
    torch.save({
        'minibatch': minibatch_i,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None
            else None,
        'loss': loss_fn.state_dict()
    },
    MODEL_PATH / exp_name / "checkpoints" / ("minibatch-%d.pt" % minibatch_i))

def loss_to_tensorboard(writer, epoch, train_loss_epoch, val_loss_epoch,
                        loss_fn):
    writer.add_scalar("Loss/train", train_loss_epoch["loss_epoch"], epoch)
    writer.add_scalar("Loss/val", val_loss_epoch["loss_epoch"], epoch)

    writer.add_scalar("Loss Task/train", train_loss_epoch["loss_task"], epoch)
    writer.add_scalar("Loss Task/val", val_loss_epoch["loss_task"], epoch)


def train(model, train_loader, val_loader, train_options, dataset_options):
    module_logger.info("Model:")
    module_logger.info(model)

    # Load options from dict
    epochs = train_options["epochs"]
    exp_name = train_options["exp_name"]
    quiet = train_options["quiet"]
    resume_from_checkpoint = train_options["resume_from_checkpoint"]

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=str(LOG_PATH / exp_name))

    # Create directories
    (MODEL_PATH / exp_name).mkdir(exist_ok=True)
    (MODEL_PATH / exp_name / "checkpoints").mkdir(exist_ok=True)

    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer(model, train_options)
    scheduler = create_lr_schedulers(optimizer, train_options)
    loss_fn = create_loss_function(train_options, dataset_options)

    module_logger.info("Loss Function:")
    module_logger.info(loss_fn)
    module_logger.info("Optimizer:")
    module_logger.info(optimizer)
    module_logger.info("Scheduler:")
    module_logger.info(scheduler)
    module_logger.info("")

    epoch_start = init_training_state(resume_from_checkpoint, model, optimizer,
                                      scheduler, loss_fn, exp_name,
                                      train_options["device"])
    if not quiet:
        print_validation_performance(model, val_loader, train_options,
                                     dataset_options, loss_fn)

    # Save initial model
    if not resume_from_checkpoint:
        torch.save(model.state_dict(),
                   MODEL_PATH / exp_name / "initial_model.pt")
        write_ex_img_to_tensorboard(train_loader, writer, train_options)

    # Initial set of masks displayed for epoch = -1
    if TB_ADD_INITIAL_MASKS and not resume_from_checkpoint:
        tensorboard_visualize_mask(
            model, writer, -1, val_loader.dataset[0][0], initial_mask=True)
    save_mask_imgs(model, MODEL_PATH / exp_name / "initial-masks")

    # Early stop critera
    early_stop_bool = False
    if train_options["early_stop_epochs"] > 0:
        module_logger.info(
            "Creating EarlyStopCriterion for %d epochs" % \
            train_options["early_stop_epochs"])
        early_stop_criterion = early_stop.EarlyStopCriterion(
            train_options["early_stop_epochs"])
    else:
        early_stop_criterion = None

    minibatch_i = 0

    for epoch in range(epoch_start, epochs):
        t_start = time.perf_counter()
        # Training
        model.train()
        train_loss_epoch, train_labels_pred, train_labels_gt, _, minibatch_i = \
            forward_pass(model, train_loader, train_options, dataset_options,
                        loss_fn, train=True, writer=writer, optimizer=optimizer,
                        scheduler=scheduler, val_loader=val_loader,
                        minibatch_i=minibatch_i)

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss_epoch, val_labels_pred, val_labels_gt, _, _ = forward_pass(
                model, val_loader, train_options, dataset_options, loss_fn,
                train=False)

        t_end = time.perf_counter()

        if epoch % train_options["checkpoint_every_n_epochs"] == 0:
            save_epoch_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                                  exp_name)

        if early_stop_criterion is not None:
            early_stop_criterion.epoch_step(epoch, val_loss_epoch["loss_epoch"])

        ## Epoch statistics
        # Performance
        if dataset_options["regression_metrics"]:
            train_rmse = rmse(train_labels_pred, train_labels_gt)
            val_rmse = rmse(val_labels_pred, val_labels_gt)
            writer.add_scalar("RMSE/train", train_rmse, epoch)
            writer.add_scalar("RMSE/val", val_rmse, epoch)
        else:
            train_acc = sklearn.metrics.accuracy_score(
                train_labels_gt.numpy(), train_labels_pred.numpy())
            val_acc = sklearn.metrics.accuracy_score(
                val_labels_gt.numpy(), val_labels_pred.numpy()
            )
            train_bal_acc = sklearn.metrics.balanced_accuracy_score(
                train_labels_gt.numpy(), train_labels_pred.numpy())
            val_bal_acc = sklearn.metrics.balanced_accuracy_score(
                val_labels_gt.numpy(), val_labels_pred.numpy()
            )
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("Balanced Accuracy/train", train_bal_acc, epoch)
            writer.add_scalar("Balanced Accuracy/val", val_bal_acc, epoch)

        # Loss
        loss_to_tensorboard(writer, epoch, train_loss_epoch, val_loss_epoch,
                            loss_fn)

        if TB_ADD_MASKS and epoch % TB_ADD_MASKS_EVERY_N == 0:
            tensorboard_visualize_mask(model, writer, epoch,
                                        val_loader.dataset[0][0])
        if not quiet and epoch % PRINT_STATISTICS_EVERY_N == 0:
            module_logger.info("Epoch: %d" % epoch)
            module_logger.info("  Train loss      : %4f" % \
                train_loss_epoch["loss_epoch"])
            module_logger.info("  Val loss        : %4f" % \
                val_loss_epoch["loss_epoch"])

            if dataset_options["regression_metrics"]:
                module_logger.info("  Train RMSE      : %.2f" % train_rmse)
                module_logger.info("  Val RMSE        : %.2f" % val_rmse)
            else:
                module_logger.info("  Train acc.      : %.2f%%" %
                                    (train_acc * 100))
                module_logger.info("  Val acc.        : %.2f%%" %
                                    (val_acc * 100))
                module_logger.info("  Train bal. acc. : %.2f%%" %
                                    (train_bal_acc * 100))
                module_logger.info("  Val bal. acc.   : %.2f%%" %
                                    (val_bal_acc * 100))

            module_logger.info("  Time            : %.2fs\n" % \
                (t_end - t_start))

        if early_stop_criterion is not None and early_stop_criterion.check_early_stop():
            early_stop_bool = True
            early_stop_epoch = epoch
            module_logger.info(
                "Stopping training early after no validation loss improvement...")
            break


    # Checkpoint final model
    if early_stop_bool:
        save_epoch_checkpoint(early_stop_epoch, model, optimizer, scheduler,
                              loss_fn, exp_name)
    elif (epochs - 1) % train_options["checkpoint_every_n_epochs"] != 0:
        save_epoch_checkpoint(epochs-1, model, optimizer, scheduler, loss_fn,
                              exp_name)


    writer.close()

def performance_metrics(labels_pred, labels_gt, dataset_options):
    """
    Compute the RMSE, accuracy, and confusion matrix for the given predictions.
    If the task is regression, accuracy and confusion matrix are none.
    If the task is classification, RMSE is none.
    """
    test_rmse = None
    test_acc = None
    test_bal_acc = None
    confusion_mat = None
    if dataset_options["regression_metrics"]:
        test_rmse = rmse(labels_pred, labels_gt)
    else:
        test_acc = sklearn.metrics.accuracy_score(
            labels_gt.numpy(), labels_pred.numpy())
        test_bal_acc = sklearn.metrics.balanced_accuracy_score(
            labels_gt.numpy(), labels_pred.numpy())
        confusion_mat = sklearn.metrics.confusion_matrix(
            labels_gt.numpy(), labels_pred.numpy(), normalize=None)

    return test_rmse, test_acc, test_bal_acc, confusion_mat

def _ensure_real_gpu(device):
    if device.type == "cuda":
        device = torch.device("cuda:%d" %
                              (device.index % torch.cuda.device_count()))
    return device

def load_model_config(run_name):
    config_path = MODEL_PATH / run_name / "configuration.pkl"
    with open(config_path, "rb") as f:
        D = pickle.load(f)
    model_options = D['model_options']
    dataset_options = D['dataset_options']
    train_options = D['train_options']

    train_options["device"] = _ensure_real_gpu(train_options["device"])

    return model_options, train_options, dataset_options

def load_model(run_name, epoch=None):
    if epoch is None:
        epoch = _most_recent_checkpoint(run_name)

    model_options, train_options, dataset_options = load_model_config(run_name)

    # Load model from the given checkpoint
    model = create_model_from_options(model_options, train_options,
                                      dataset_options)
    # Load from saved-checkpoints/ if the epoch exists there
    if epoch is not None:
        if isinstance(epoch, int):
            saved_checkpoint_path = \
                MODEL_PATH / train_options["exp_name"] / "saved-checkpoints" / \
                ("%d.pt" % epoch)
        else:
            saved_checkpoint_path = \
                MODEL_PATH / train_options["exp_name"] / "saved-checkpoints" / \
                ("%s.pt" % epoch)

    if saved_checkpoint_path.exists():
        load_checkpoint_path = saved_checkpoint_path
    else:
        if isinstance(epoch, int):
            load_checkpoint_path = MODEL_PATH / train_options["exp_name"] / \
                "checkpoints" / ("%d.pt" % epoch)
        else:
            load_checkpoint_path = MODEL_PATH / train_options["exp_name"] / \
                "checkpoints" / ("%s.pt" % epoch)

    D = torch.load(load_checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(D['model'], strict=True)

    return model, model_options, train_options, dataset_options

def eval_model(name, epoch):
    """
    Evaluate the model on the test set at the given checkpoint. The test results
    are saved to the model's directory and reused if the same checkpoint is
    queried again.
    """
    def _checkpoint_path(name, epoch):
        if isinstance(epoch, str):
            path = MODEL_PATH / name / "checkpoints" / ("%s.pt" % epoch)
        else:
            path = MODEL_PATH / name / "checkpoints" / ("%d.pt" % epoch)
        return path

    def _saved_checkpoint_path(name, epoch):
        if isinstance(epoch, str):
            path = MODEL_PATH / name / "saved-checkpoints" / ("%s.pt" % epoch)
        else:
            path = MODEL_PATH / name / "saved-checkpoints" / ("%d.pt" % epoch)
        return path

    def _results_path(name, epoch):
        if isinstance(epoch, str):
            path = MODEL_PATH / name / "results" / ("%s.pt" % epoch)
        else:
            path = MODEL_PATH / name / "results" / ("%d.pt" % epoch)
        return path

    def _results_exist(name, epoch):
        return _results_path(name, epoch).exists()

    def _save_results(name, epoch, results):
        path = _results_path(name, epoch)
        path.parent.mkdir(exist_ok=True)
        torch.save(results, path)

    def _load_results(name, epoch):
        assert _results_exist(name, epoch)
        path = _results_path(name, epoch)
        res = torch.load(path, map_location=torch.device("cpu"))
        return res

    def _checkpoint_saved(name, epoch):
        path = _saved_checkpoint_path(name, epoch)
        return path.exists()

    def _save_checkpoint(name, epoch):
        orig_path = _checkpoint_path(name, epoch)
        dest_path = _saved_checkpoint_path(name, epoch)
        dest_path.parent.mkdir(exist_ok=True)
        shutil.copy(orig_path, dest_path)

    if not _checkpoint_saved(name, epoch):
        _save_checkpoint(name, epoch)

    if _results_exist(name, epoch):
        return _load_results(name, epoch)
    else:
        model, model_options, train_options, dataset_options = \
            load_model(name, epoch)
        test_loader = datasets.load_test_data(model_options, dataset_options,
                                              train_options["device"])
        test_results = \
            test_single_model(model, test_loader, train_options,
                              dataset_options)
        _save_results(name, epoch, test_results)

    return test_results


@torch.no_grad()
def test_single_model(model, test_loader, train_options, dataset_options):
    model.eval()

    loss_fn = create_loss_function(train_options, dataset_options)

    # Forward pass on the test set
    test_loss, labels_pred, labels_gt, logits_pred, _ = forward_pass(
        model, test_loader, train_options, dataset_options, loss_fn,
        train=False)
    test_rmse, test_acc, test_bal_acc, confusion_mat = performance_metrics(
        labels_pred, labels_gt, dataset_options
    )

    test_results = {
        "test_loss": test_loss,
        "labels_pred": labels_pred,
        "labels_gt": labels_gt,
        "logits_pred": logits_pred,
        "test_rmse": test_rmse,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "confusion_mat": confusion_mat,
    }

    return test_results


def parse_args():
    parser = argparse.ArgumentParser(description="Mincam Optimizer")

    parser.add_argument("-n", "--num_freeform_pixels", type=int, default=1,
        help="Number of freeform pixels")

    parser.add_argument("--img_r", type=int, default=200,
                        help="Image size (rows)")
    parser.add_argument("--img_c", type=int, default=300,
                        help="Image size (cols)")

    parser.add_argument("-e", "--epochs", type=int, default=1,
        help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=512,
        help="GPU minibatch size")
    parser.add_argument("--gpu_dataset", action="store_true",
        help="Store the entire dataset on the GPU")
    parser.add_argument("-d", "--device_id", type=str,
        default="cuda:0", help="PyTorch device name")
    parser.add_argument("--num_workers", type=int, default=0,
        help="Number of dataloader wokers")
    parser.add_argument("--lr", type=float, default=1e-2,
        help="Learning rate")
    parser.add_argument("--lr_scheduler", action='store_true', default=False,
        help="Use learning rate scheduler")
    parser.add_argument("--resume_from_checkpoint", action='store_true',
        default=False, help="""Resume training from last
        checkpoint""")
    parser.add_argument("--quiet", action='store_true', default=False,
        help="Do not print training messages")
    parser.add_argument("--base_exp_name", type=str,
        default=str(datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")),
        help="Base experiment name")
    parser.add_argument("--model_type", type=str, default=None,
        help="Model type (baseline or mincam)")

    parser.add_argument("--log", type=str, default="info",
                        help="Log level (debug, info, warning, error)")

    args = parser.parse_args()

    if args.gpu_dataset and args.num_workers > 0:
        module_logger.error(
            "Cannot use num_workers > 0 when the dataset is on the GPU")
        sys.exit(1)

    return args

def create_model_from_options(model_options, train_options, dataset_options):

    # Extract variables from dictionaries
    device = train_options["device"]
    regression_loss = train_options["regression_loss"]
    model_type = model_options["model_type"]
    num_classes = dataset_options["num_classes"]
    img_size = dataset_options["img_size"]
    hidden_layer_size = model_options["hidden_layer_size"]
    hidden_layer_count = model_options["hidden_layer_count"]
    mask_init_method = model_options["mask_init_method"]

    if regression_loss:
        output_size = 1
    else:
        output_size = num_classes

    model_kwargs = {
        "output_size": output_size,
        "img_size": img_size,
        "hidden_layer_size": hidden_layer_size,
        "hidden_layer_count": hidden_layer_count,
    }

    if model_type == "baseline":
        model_kwargs = {**model_kwargs}
        model = camera_models.Baseline(**model_kwargs).to(device)

    elif model_type == "mincam":
        mincam_kwargs = {
            "mincam_size": model_options["mincam_size"],
            "realistic_sensor": model_options["mincam_realistic_sensor"],
            "sensor_gain": model_options["mincam_sensor_gain"],
            "sensor_n_bits": model_options["mincam_sensor_n_bits"],
            "sensor_saturation_val":
                model_options["mincam_sensor_saturation_val"],
            "read_noise_std": model_options["mincam_read_noise_std"],
            "mask_init_method": mask_init_method,
            "simulate_pd_area_blur": model_options["simulate_pd_area_blur"],
            "mask_blur_kernel_sigma": model_options["mask_blur_kernel_sigma"],
            "simulate_directivity": model_options["simulate_directivity"],
            "mask_min_value": model_options["mask_min_value"],
            "mask_max_value": model_options["mask_max_value"],
            "model_vert_fov": model_options["model_vert_fov"],
            "model_horiz_fov": model_options["model_horiz_fov"],
        }
        model_kwargs = {**model_kwargs, **mincam_kwargs}

        model = camera_models.Mincam(**model_kwargs).to(device)

    else:
        module_logger.error("Model type not supported: %s" % model_type)
        sys.exit(1)

    if isinstance(model, camera_models.Mincam):
        assert not \
            torch.any(torch.isnan(model._mask_param_transform(model.masks)))

    if "initial_masks" in train_options.keys():
        initial_masks = train_options["initial_masks"] # type:torch.Tensor
        with torch.no_grad():
            model.masks[:] = initial_masks.to(device)
            assert not torch.any(torch.isnan(model.masks))

    return model


def pprint_config(model_options, train_options, dataset_options):
    pp = pprint.PrettyPrinter(indent=4)

    module_logger.info("Model Options")
    module_logger.info(pp.pformat(model_options))
    module_logger.info("")

    module_logger.info("Train Options")
    module_logger.info(pp.pformat(train_options))
    module_logger.info("")

    module_logger.info("Dataset Options")
    module_logger.info(pp.pformat(dataset_options))
    module_logger.info("")

def train_single_model(train_dataloader, val_dataloader, model_options,
                       train_options, dataset_options):

    # Print configuration
    pprint_config(model_options, train_options, dataset_options)

    model = create_model_from_options(model_options, train_options,
                                      dataset_options)

    train(model, train_dataloader, val_dataloader, train_options,
          dataset_options)


def save_configuration_to_file(model_options, train_options, dataset_options):
    # Save options to file
    path = os.path.join(MODEL_PATH, train_options["exp_name"],
                        "configuration.pkl")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    configuration_dict = {
        "model_options": model_options,
        "train_options": train_options,
        "dataset_options": dataset_options
    }

    with open(path, 'wb') as f:
        pickle.dump(configuration_dict, f)

def build_exp_name(model_options, dataset_options):
    if model_options["model_type"] in ["baseline", "conv"]:
        return "%s-%dx%d_%s" % \
            (model_options["base_exp_name"],
             dataset_options["img_size"][0], dataset_options["img_size"][1],
             model_options["model_type"])
    else:
        return "%s-m%d_%s" % \
            (model_options["base_exp_name"],
             model_options["mincam_size"],
             model_options["model_type"])

def model_config_from_args():
    args = parse_args()
    device = torch.device(args.device_id)

    # Create configuration dictionaries
    dataset_options = {
        "gpu_dataset": args.gpu_dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "shuffle": True,
        "img_size": (args.img_r, args.img_c),
    }

    model_options = {
        "base_exp_name": args.base_exp_name,
        "model_type": args.model_type,
        "hidden_layer_size": 128,
        "hidden_layer_count": 2,
        "mincam_realistic_sensor": True,
        "mincam_sensor_gain": 0.0012243347290119763,
        "mincam_sensor_n_bits": 12,
        "mincam_sensor_saturation_val": 1,
        "mincam_read_noise_std": 250e-6,
        "mask_init_method": "random",
        "simulate_pd_area_blur": False,
        "mask_blur_kernel_sigma": 0,
        "simulate_directivity": False,
        "mask_min_value": 0,
        "mask_max_value": 1,
        "model_vert_fov": 70,
        "model_horiz_fov": 70
    }

    if model_options["model_type"] == "mincam":
        model_options["mincam_size"] = args.num_freeform_pixels

    train_options = {
        "epochs": args.epochs,
        "lr": args.lr,
        "device": device,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "quiet": args.quiet,
        "lr_scheduler": args.lr_scheduler,
        "warmup_lr": False,
        "augmentation_fn": datasets.DataAugmentation(),
        "regression_loss": False,
        "early_stop_epochs": -1,
        "checkpoint_every_n_epochs": 100,
        "checkpoint_minibatches_until": 0
    }

    numeric_log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)])

    return dataset_options, model_options, train_options

def load_training_data(model_options, train_options, dataset_options):
    # Load dataset from labeled video
    train_device = train_options["device"]
    train_dataloader, val_dataloader, num_classes = \
        datasets.load_train_val_data(model_options, dataset_options,
                                     train_device)
    dataset_options["num_classes"] = num_classes

    return train_dataloader, val_dataloader

def create_loss_function(train_options, dataset_options) -> loss.Loss:
    # Set the loss function
    if train_options["regression_loss"]:
        task_loss = nn.MSELoss(reduction="none")
    else:
        task_loss = nn.CrossEntropyLoss(reduction="none")

    return loss.Loss(task_loss).to(train_options["device"])

def _validate_options(model_options, train_options, dataset_options):
    assert model_options["model_type"] in \
        ["baseline", "mincam", "mincam_binary", "mincam_fixed_mask",
         "mincam_refine", "conv"]

def run(model_options, train_options, dataset_options, train_dataloader,
        val_dataloader):
    """
    Train and test a single model defined by the configuration dictionaries
    using the given dataloaders
    """
    _validate_options(model_options, train_options, dataset_options)

    # Create the full model name
    train_options["exp_name"] = build_exp_name(model_options, dataset_options)

    # Save configuration
    save_configuration_to_file(model_options, train_options, dataset_options)

    # Training
    train_single_model(train_dataloader, val_dataloader, model_options,
                       train_options, dataset_options)


def init_libraries():
    random_seed = 20220920
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4, sci_mode=False)

    # TensorFloat32 precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    np.random.seed(random_seed)
    np.set_printoptions(precision=4)

def main():
    init_libraries()

    dataset_options, model_options, train_options = model_config_from_args()

    dataset_options["label_select"] = "people_count"
    dataset_options["regression_metrics"] = dataset_options["label_select"] == "people_count"
    dataset_options["dataset_name"] = "toy-example"

    # Load data
    train_dataloader, val_dataloader = \
        load_training_data(model_options, train_options, dataset_options)

    # Training
    run(model_options, train_options, dataset_options, train_dataloader,
        val_dataloader)


if __name__ == "__main__":
    main()
