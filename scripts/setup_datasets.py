import os
import sys
import argparse
import requests
import zipfile
import io
import shutil
import torch
import torchvision.datasets as datasets
from PIL import Image
import numpy as np

def setup_cifar100(data_dir='./data', mock=False):
    print("Setting up CIFAR-100...")
    os.makedirs(data_dir, exist_ok=True)
    if mock:
        print("Creating mock CIFAR-100 data folder...")
        mock_file = os.path.join(data_dir, 'cifar-100-mock-completed')
        with open(mock_file, 'w') as f:
            f.write("mock")
        print("Mock CIFAR-100 setup complete.")
        return
        
    try:
        datasets.CIFAR100(root=data_dir, train=False, download=True)
        print("CIFAR-100 setup complete.")
    except Exception as e:
        print(f"Error downloading CIFAR-100: {e}. Run with --mock for offline mock generation.")

def setup_tiny_imagenet(data_dir='./data', mock=False):
    print("Setting up Tiny-ImageNet...")
    target_path = os.path.join(data_dir, 'tiny-imagenet-200')
    if os.path.exists(target_path):
        print("Tiny-ImageNet already exists.")
        return
        
    os.makedirs(data_dir, exist_ok=True)
    if mock:
        print("Creating mock Tiny-ImageNet folders...")
        # Create a toy directory structure
        val_dir = os.path.join(target_path, 'val')
        os.makedirs(val_dir, exist_ok=True)
        # Create a mock class folder and copy a dummy image
        mock_class = os.path.join(val_dir, 'n01440750')
        os.makedirs(mock_class, exist_ok=True)
        
        # Save a 64x64 dummy image
        img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
        img.save(os.path.join(mock_class, 'val_0.JPEG'))
        print("Mock Tiny-ImageNet setup complete.")
        return

    # Real CS231n download
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    try:
        print(f"Downloading Tiny-ImageNet from {url}...")
        response = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        print("Extracting Tiny-ImageNet...")
        z.extractall(data_dir)
        
        # Fix validation annotations
        from src.datasets.tiny_imagenet_setup import fix_tiny_imagenet_val_folder
        fix_tiny_imagenet_val_folder(target_path)
        print("Tiny-ImageNet setup complete.")
    except Exception as e:
        print(f"Error setting up Tiny-ImageNet: {e}. Falling back to mock dataset setup.")
        setup_tiny_imagenet(data_dir, mock=True)

def setup_imagenet(data_dir='./data', mock=False):
    print("Setting up ImageNet...")
    imagenet_dir = os.path.join(data_dir, 'imagenet')
    val_dir = os.path.join(imagenet_dir, 'val')
    
    if os.path.exists(val_dir):
        print("ImageNet validation folder already exists.")
        return
        
    os.makedirs(val_dir, exist_ok=True)
    
    if mock:
        print("Creating mock ImageNet validation folders (10 classes)...")
        mock_classes = ['n01440750', 'n01443537', 'n01491361', 'n01531178', 'n01558990', 
                        'n01614990', 'n01629819', 'n01641577', 'n01644332', 'n01644900']
        for cls in mock_classes:
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # Create a 224x224 mock image
            img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            img.save(os.path.join(cls_dir, 'mock_val_0.JPEG'))
        print("Mock ImageNet setup complete.")
        return
        
    print("ImageNet dataset is extremely large (150GB+). Setting up a sample ImageNet subset...")
    # Generate mock subset for validation runs if real download is not provided
    setup_imagenet(data_dir, mock=True)

def main():
    parser = argparse.ArgumentParser(description="Download and setup datasets (CIFAR-100, Tiny-ImageNet, ImageNet)")
    parser.add_argument("--dataset", type=str, default="all", choices=["cifar100", "tiny_imagenet", "imagenet", "all"],
                        help="The dataset to setup or 'all'")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset storage")
    parser.add_argument("--mock", action="store_true", help="Perform mock setup for fast offline testing")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ["cifar100", "all"]:
        setup_cifar100(args.data_dir, args.mock)
    if args.dataset in ["tiny_imagenet", "all"]:
        setup_tiny_imagenet(args.data_dir, args.mock)
    if args.dataset in ["imagenet", "all"]:
        setup_imagenet(args.data_dir, args.mock)

    print("Setup processes completed successfully!")

if __name__ == "__main__":
    main()
