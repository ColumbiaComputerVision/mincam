import sys
import numpy as np
import torch
import torch.nn as nn
import torch.jit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pickle
import torchvision.transforms.functional as F_t
from torch.utils.data import TensorDataset
from typing import List

import constants

module_logger = logging.getLogger(__name__)

class DataAugmentation(nn.Module):
    @torch.no_grad()
    def forward(self, x):
        assert x.ndim == 4 or x.ndim == 2 # 4: image, 2: precomputed measurement

        # Random gain ~ Uniform(0.8, 1.2)
        g = torch.rand(x.shape[0], device=x.device) * 0.4 + 0.8

        if x.ndim == 4:
            x = x * g[:,None,None,None]
        elif x.ndim == 2:
            x = x * g[:,None]
        else:
            module_logger.error("Dimension error")
            sys.exit(1)

        return x


class FrameDataset(Dataset):
    """
    Image dataset where frames is a tensor stored in memory
    """
    def __init__(self, frames, labels, dataset_device, shuffle=False, pin_memory=False):
        super().__init__()

        if type(frames) == np.ndarray:
            frames = torch.from_numpy(frames)
        if type(labels) == np.ndarray:
            labels = torch.from_numpy(labels)

        if shuffle:
            # Create a shuffled copy of the dataset
            frames_shuffle, labels_shuffle = self._shuffle(frames, labels)
            frames = frames_shuffle
            labels = labels_shuffle

        self.frames = frames[:,None,:,:]
        self.labels = labels

        self.frames = self.frames.to(dataset_device)
        self.labels = self.labels.to(dataset_device)

        if pin_memory:
            assert dataset_device == torch.device("cpu")
            self.frames = self.frames.pin_memory()
            self.labels = self.labels.pin_memory()

        # Compute the number of classes
        if self.labels.ndim == 1:
            self.num_classes = len(self.labels.unique())
        elif self.labels.ndim == 2:
            self.num_classes = self.labels.shape[1]
        else:
            module_logger.error("Labels must be 1- or 2-dimensional")
            sys.exit(1)

        self.img_size = frames.shape[2:]

        assert self.frames.shape[0] == self.labels.shape[0]

    def _shuffle(self, frames, labels):
        """
        Return a shuffled copy of the frames and labels
        """
        shuffle_idx = torch.randperm(
            frames.shape[0], generator=torch.Generator().manual_seed(51823))

        frames_shuffled = frames[shuffle_idx].contiguous()
        labels_shuffled = labels[shuffle_idx].contiguous()

        return frames_shuffled, labels_shuffled

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, index):
        x = self.frames[index]
        if x.dtype == torch.uint8: # cast to float on-the-fly if stored as uint8
            x = x.to(torch.get_default_dtype()).div(255)
        y = self.labels[index]

        return x, y

class FrameLoader:
    def __init__(self, frame_dataset, batch_size, drop_last) -> None:
        assert isinstance(frame_dataset, FrameDataset) or \
            isinstance(frame_dataset, TensorDataset)

        self.dataset = frame_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if not drop_last:
            self.num_batches = int(np.ceil(len(frame_dataset) / batch_size))
        else:
            self.num_batches = len(frame_dataset) // batch_size

        self.batch_i = 0

    def __iter__(self):
        self.batch_i = 0
        return self

    def __next__(self):
        if self.batch_i == self.num_batches:
            raise StopIteration

        i = self.batch_i * self.batch_size

        if not self.drop_last and self.batch_i == self.num_batches - 1:
            end_i = len(self.dataset)
        else:
            end_i = i + self.batch_size

        self.batch_i += 1

        return self.dataset[i:end_i]

    def __len__(self):
        return self.num_batches


class ImageFolderDataset(Dataset):
    def __init__(self, path: Path, label_select: str):
        super().__init__()

        with open(path / "labels.pkl", "rb") as f:
            labels = pickle.load(f)[label_select] # type:torch.Tensor

        self.labels = labels
        # Compute the number of classes
        if self.labels.ndim == 1:
            self.num_classes = len(self.labels.unique())
        elif self.labels.ndim == 2:
            self.num_classes = self.labels.shape[1]
        else:
            module_logger.error("Labels must be 1- or 2-dimensional")
            sys.exit(1)

        self.N = self.labels.shape[0]

        self.img_path = path / "imgs"

        self.loader = self._load_img_pt

        x, _ = self[0]
        self.img_size = x.shape

    def _load_img_pt(self, index):
        path = self.img_path / ("%d.pt" % index)
        img = torch.load(path)
        img = F_t.convert_image_dtype(img, torch.get_default_dtype())
        if img.ndim == 2:
            img = img[None,:,:]

        return img

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = self.loader(index) # CxHxW
        y = self.labels[index]

        return x, y



def dataset_from_file(path, label_select, dataset_device, shuffle,
                      pin_memory):
    """
    Create a FrameDataset using the images and labels stored at the given path.
    """
    D = torch.load(path, weights_only=False)
    imgs = D["imgs"]
    labels = D["labels"][label_select]
    dataset = FrameDataset(imgs, labels, dataset_device, shuffle, pin_memory)
    return dataset

def dataset_from_folder(path, label_select):
    """
    Create a ImageFolderDataset using the images and labels at the given path.
    """
    return ImageFolderDataset(path, label_select)

def create_dataloader_pytorch(dataset, dataset_options, batch_size, shuffle,
                              drop_last, dataset_device):
    """
    Create a dataloader for the given dataset
    """
    num_workers = dataset_options["num_workers"]
    pin_memory = dataset_device == torch.device("cpu")
    persistent_workers = num_workers > 0

    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        drop_last=drop_last)

    return dataloader

def create_dataloader_synchronous(dataset, batch_size, drop_last):
    """
    Create a dataloader for the given dataset
    """
    dataloader = FrameLoader(dataset, batch_size, drop_last)

    return dataloader

def _is_file_dataset(dataset_path: Path, dnames: List[str]):
    for dname in dnames:
        if not (dataset_path / (dname + ".pt")).exists():
            return False
    return True

def processed_dataset_path_from_dname(dataset_name, img_size):
    if dataset_name == "toy-example":
        return constants.DATASET_PATH / "toy-example" / ("%dx%d" % img_size)
    else:
        module_logger.error("Dataset %s not supported." % dataset_name)
        sys.exit(1)

def load_train_val_data(model_options, dataset_options, train_device):
    """
    Main function to load the training and validation datasets and return
    dataloaders.
    """
    img_size = dataset_options["img_size"]
    label_select = dataset_options["label_select"]
    dataset_device = train_device \
        if dataset_options["gpu_dataset"] else torch.device("cpu")
    dataset_name = dataset_options["dataset_name"]

    processed_dataset_path = processed_dataset_path_from_dname(
        dataset_name, img_size)
    if not processed_dataset_path.exists():
        module_logger.error("Processed data does not exist for dataset")
        sys.exit(1)

    # Dataset options
    train_shuffle = dataset_options["shuffle"]
    val_shuffle = False
    train_drop_last = True
    val_drop_last = False
    train_batch_size = dataset_options["batch_size"]
    val_batch_size = dataset_options["batch_size"]
    pin_memory = dataset_device == torch.device("cpu") and \
        train_device != torch.device("cpu")


    if _is_file_dataset(processed_dataset_path, ["train", "val"]):
        # Load dataset from train.pt, val.pt
        train_path = processed_dataset_path / "train.pt"
        val_path = processed_dataset_path / "val.pt"

        # Create datasets
        train_set = dataset_from_file(
            train_path, label_select, dataset_device, train_shuffle, pin_memory)
        val_set = dataset_from_file(
            val_path, label_select, dataset_device, val_shuffle, pin_memory)

        # Create dataloaders
        assert dataset_options["num_workers"] == 0
        train_loader = create_dataloader_synchronous(
            train_set, train_batch_size, train_drop_last)
        val_loader = create_dataloader_synchronous(
            val_set, val_batch_size, val_drop_last)

    else:
        assert not dataset_options["gpu_dataset"]
        # Create dataset with images in a folder
        train_path = processed_dataset_path / "train"
        val_path = processed_dataset_path / "val"

        train_set = dataset_from_folder(train_path, label_select)
        val_set = dataset_from_folder(val_path, label_select)

        train_loader = create_dataloader_pytorch(
            train_set, dataset_options, train_batch_size, train_shuffle,
            train_drop_last, dataset_device)

        val_loader = create_dataloader_pytorch(
            val_set, dataset_options, val_batch_size, val_shuffle,
            val_drop_last, dataset_device)


    module_logger.info("Training set")
    print_dataset_statistics(train_set)
    module_logger.info("Validation set")
    print_dataset_statistics(val_set)

    # Check the classes are the same in the training and validation sets.
    if isinstance(train_set, TensorDataset):
        train_n_classes = len(torch.unique(train_set.tensors[1]))
        val_n_classes = len(torch.unique(val_set.tensors[1]))
    else:
        train_n_classes = train_set.num_classes
        val_n_classes = val_set.num_classes
    if train_n_classes != val_n_classes:
        module_logger.warn(
            "Training and validation set contain a different number of classes.")

    return train_loader, val_loader, train_n_classes

def load_test_data(model_options, dataset_options, inference_device):
    """
    Load the test data
    """
    img_size = dataset_options["img_size"]
    label_select = dataset_options["label_select"]
    dataset_device = inference_device \
        if dataset_options["gpu_dataset"] else torch.device("cpu")
    dataset_name = dataset_options["dataset_name"]

    processed_dataset_path = processed_dataset_path_from_dname(
        dataset_name, img_size)
    if not processed_dataset_path.exists():
        module_logger.error("Processed data does not exist for dataset")
        sys.exit(1)

    # Create dataset
    test_shuffle = False
    test_batch_size = dataset_options["batch_size"]
    test_drop_last = False
    pin_memory = dataset_device == torch.device("cpu") and \
        inference_device != torch.device("cpu")


    if _is_file_dataset(processed_dataset_path, ["test"]):
        test_path = processed_dataset_path / "test.pt"

        test_set = dataset_from_file(test_path, label_select, dataset_device,
                                     test_shuffle, pin_memory)

        # Create dataloader
        test_loader = create_dataloader_synchronous(
            test_set, test_batch_size, test_drop_last)
    else:
        assert not dataset_options["gpu_dataset"]
        # Create dataset with images in a folder
        test_path = processed_dataset_path / "test"

        test_set = dataset_from_folder(test_path, label_select)

        test_loader = create_dataloader_pytorch(
            test_set, dataset_options, test_batch_size, test_shuffle,
            test_drop_last, dataset_device)


    return test_loader


def print_dataset_statistics(dataset):
    """
    Iterate over the dataset to display class statistics
    """
    assert isinstance(dataset, FrameDataset) or \
        isinstance(dataset, ImageFolderDataset) or \
        isinstance(dataset, TensorDataset)

    if isinstance(dataset, FrameDataset) or isinstance(dataset, ImageFolderDataset):
        y = dataset.labels.clone()
    elif isinstance(dataset, TensorDataset):
        y = dataset.tensors[1].clone()

    if y.ndim > 1: # Argmax gives classes if y is a PDF
        y = y.argmax(1)

    classes, counts = y.unique(sorted=True, return_counts=True)
    classes = classes.cpu()
    counts = counts.cpu()

    for i in range(len(classes)):
        module_logger.info("%d: N = %d" % (classes[i], counts[i]))
    module_logger.info("\n")

