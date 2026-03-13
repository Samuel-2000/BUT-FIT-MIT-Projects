from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.models import resnet18
from . import paths

import shutil
import requests
import zipfile

import random


def download_cityscapes():
    # URL for the login action
    login_url = 'https://www.cityscapes-dataset.com/login/'

    # URLs for downloading the file after logging in
    urls = [
        'https://www.cityscapes-dataset.com/file-handling/?packageID=3',
        'https://www.cityscapes-dataset.com/file-handling/?packageID=7'
    ]

    # login credentials and any other required form data for POST request
    login_data = {
        'username': 'bahno',
        'password': 'X4W?y2H%h-t445t',
        'submit': 'Login'
    }

    # start a session so that cookies are saved between requests
    with requests.Session() as session:
        # send POST request to the login URL with the login data
        response = session.post(login_url, data=login_data)

        # check if login was successful
        if response.status_code == 200:
            print("Login successful")
        else:
            print("Login failed")
            exit()

        # send GET request to the download URLs
        for url in urls:
            print(f"Fetching file from: {url}")
            response = session.get(url, stream=True)

            if response.status_code != 200:
                print(f"Dataset download error: {response.status_code}")
                return

            # write the content to a file
            content_disposition = response.headers.get('content-disposition')
            filename = content_disposition.split('filename=')[1].replace("\"", "")
            directory_to_extract_to = paths.image
            if "disparity" in filename:
                directory_to_extract_to = paths.depth

            file_path = os.path.join(directory_to_extract_to, filename)

            # file not yet downloaded
            if not os.path.exists(directory_to_extract_to):
                print(f"Downloading file from: {url}")
                os.makedirs(directory_to_extract_to, exist_ok=True)  # make dirs if not exist
                with open(file_path, 'wb') as download_file:
                    for chunk in response.iter_content(chunk_size=1048576):
                        download_file.write(chunk)

                print(f"File downloaded successfully as {file_path}")

                # unzip contents
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # extract the contents
                    zip_ref.extractall(directory_to_extract_to)
                    zip_ref.close()

            # remove old zip file, license and readme
            if os.path.exists(file_path):
                os.remove(file_path)

            if os.path.exists(os.path.join(directory_to_extract_to, "license.txt")):
                os.remove(os.path.join(directory_to_extract_to, "license.txt"))

            if os.path.exists(os.path.join(directory_to_extract_to, "README")):
                os.remove(os.path.join(directory_to_extract_to, "README"))

            # parent folder to be removed
            core_folder = os.path.join(directory_to_extract_to, filename.split("_")[0])

            # move the contents of the parent folder to its original place
            for item in os.listdir(core_folder):
                item_path = os.path.join(core_folder, item)

                # check if the item is a folder
                if os.path.isdir(item_path):
                    # construct the destination path for the current item
                    destination_item_path = os.path.join(directory_to_extract_to, item)
                    # move the subfolder to the destination
                    shutil.move(item_path, destination_item_path)

            # remove the parent folder
            os.rmdir(core_folder)



class CityscapesDataset(Dataset):
    def __init__(self, depth_dir: str, data_dir: str, resnet_weights: dict = resnet18(pretrained=False).state_dict()):
        self.depth_dir = depth_dir
        self.data_dir = data_dir # /image
        self.resnet_weights = resnet_weights
        self.preprocess = Compose([
            CenterCrop((896, 1792)),
            Resize((224, 448)),
            ToTensor()
        ])
        self.preprocess_2x = Compose([
            CenterCrop((896, 1792)),
            Resize((448, 896)),
            ToTensor()
        ])
        self.preprocess_4x = Compose([
            CenterCrop((896, 1792)),
            #Resize((896, 1792)),
            ToTensor()
        ])
        self.all_images = []
        # Traverse through all subdirectories and their files
        for root, dirs, files in os.walk(self.data_dir):
            self.all_images.extend([os.path.join(root, file) for file in files])

        self.all_depths = []
        # Traverse through all subdirectories and their files
        for root, dirs, files in os.walk(self.depth_dir):
            self.all_depths.extend([os.path.join(root, file) for file in files])


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.all_images):
            raise IndexError("Index out of range")
        
        # Select the image file based on idx
        image_path = self.all_images[idx]
        depth_path = self.all_depths[idx]

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)

        #size_index = random.randint(0, 2)
        size_index = 0

        if size_index == 0:
            image = self.preprocess(image)
            depth = self.preprocess(depth)
        elif size_index == 1:
            image = self.preprocess_2x(image)
            depth = self.preprocess_2x(depth)
        else:
            image = self.preprocess_4x(image)
            depth = self.preprocess_4x(depth)


        return image.float(), depth.float()

