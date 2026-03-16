#!/bin/bash

# A more robust downloader for Tiny ImageNet from a fast mirror (academic source)
# Mirror from Geifman et al. (2019) or similar
mkdir -p Image/data
cd Image/data

echo "[*] Attempting to download Tiny ImageNet..."
# Using a common mirror if stanford is down
curl -L -o tiny-imagenet-200.zip https://github.com/TengdaHan/Tiny-ImageNet/releases/download/v1.0/tiny-imagenet-200.zip

if [ $? -eq 0 ]; then
    echo "[*] Download successful. Extracting..."
    unzip -q tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
    
    # Fix the val folder structure (Tiny ImageNet standard issue)
    echo "[*] Fixing validation folder structure..."
    cd tiny-imagenet-200/val
    mkdir -p images_fixed
    while read line; do
        fname=$(echo $line | awk '{print $1}')
        cname=$(echo $line | awk '{print $2}')
        mkdir -p "$cname"
        mv "images/$fname" "$cname/"
    done < val_annotations.txt
    rm -rf images
    echo "[*] Tiny ImageNet setup complete."
else
    echo "[!] Download failed. Please download tiny-imagenet-200.zip manually to Image/data/"
fi
