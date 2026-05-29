import os
import requests
import zipfile
import io
import shutil

def download_tiny_imagenet(data_dir='./data'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    target_path = os.path.join(data_dir, 'tiny-imagenet-200')
    
    if os.path.exists(target_path):
        print("Tiny ImageNet already exists.")
        return
    
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading Tiny ImageNet from {url}...")
    response = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    print("Extracting...")
    z.extractall(data_dir)
    print("Tiny ImageNet setup complete.")

def fix_tiny_imagenet_val_folder(data_dir='./data/tiny-imagenet-200'):
    """
    Tiny ImageNet validation folder has all images in one folder.
    This script moves them into subfolders based on val_annotations.txt
    to match standard ImageFolder format.
    """
    val_dir = os.path.join(data_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annot_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if not os.path.exists(val_annot_file):
        print("Validation annotations not found or already processed.")
        return

    print("Formatting Tiny ImageNet validation folder...")
    with open(val_annot_file, 'r') as f:
        for line in f.readlines():
            parts = line.split('\t')
            img_file = parts[0]
            cls_id = parts[1]
            
            cls_dir = os.path.join(val_dir, cls_id)
            os.makedirs(cls_dir, exist_ok=True)
            
            src_path = os.path.join(val_img_dir, img_file)
            dst_path = os.path.join(cls_dir, img_file)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
    
    # Remove empty images folder and annotation file
    if os.path.exists(val_img_dir):
        shutil.rmtree(val_img_dir)
    os.remove(val_annot_file)
    print("Tiny ImageNet validation folder formatted.")

if __name__ == "__main__":
    download_tiny_imagenet()
    fix_tiny_imagenet_val_folder()
