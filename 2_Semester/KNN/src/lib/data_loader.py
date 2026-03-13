import torch.utils.data
from . import paths
from torch import utils as utils
from torch.utils.data import Subset
from .cityscapes import CityscapesDataset
from .cityscapes import download_cityscapes
import os


def data_load(args):
    """
    Function prepares and loads data
    :param args: dataset name enum arg
    :return: train loader, test loader
    """
    if args.dataset == 0:
        # Instantiate CityscapesDataset
        if not (paths.depth.exists()):
            download_cityscapes()

        cityscapes_dataset = CityscapesDataset(paths.depth, paths.image)

        # Dictionary to store the count of images in each subfolder
        subfolder_counts = {}

        # Calculate the count of images in each subfolder
        for image_path in cityscapes_dataset.all_images:
            subfolder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))  # Extract the directory name
            subfolder_counts[subfolder] = subfolder_counts.get(subfolder, 0) + 1

        train_count = subfolder_counts.get("train", 0)
        print(f"train_count: {train_count}")
        train_set = Subset(cityscapes_dataset, range(train_count))

        test_count = subfolder_counts.get("test", 0)
        print(f"test_count: {test_count}")
        test_set = Subset(cityscapes_dataset, range(train_count, train_count + test_count))

        val_count = subfolder_counts.get("val", 0)
        print(f"val_count: {val_count}")
        val_set = Subset(cityscapes_dataset, range(train_count + test_count, train_count + test_count + val_count))


    else:
        print("error - zly argument dataset")
        exit(0)

    
    #tr = range(1000)
    te = range(1000)
    #val = range(100)
    
        # chooses only subsety of data (performance)
    #train_set = torch.utils.data.Subset(train_set, tr)
    test_set = torch.utils.data.Subset(test_set, te)
    #val_set = torch.utils.data.Subset(val_set, val)
    
    # get data loaders
    train_loader = utils.data.DataLoader(train_set, 1, shuffle=True) #True
    test_loader = utils.data.DataLoader(test_set, 1, shuffle=False)
    val_loader = utils.data.DataLoader(val_set, 1, shuffle=False)

    return train_loader, test_loader, val_loader, test_set[0]

