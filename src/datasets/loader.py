import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_tiny_imagenet(batch_size=128, data_dir='./data', resize=None):
    transform_list = [transforms.ToTensor()]
    if resize:
        transform_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform_list)
    
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    if not os.path.exists(val_dir):
        print(f"Warning: Tiny ImageNet not found at {val_dir}. Hãy chạy datasets/tiny_imagenet_setup.py trước.")
        return None
        
    val_set = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader

def get_cifar10(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Warning: Could not load CIFAR10: {e}. Falling back to random mock dataset for offline execution.")
        import torch
        from torch.utils.data import TensorDataset
        # Create 128 mock images of CIFAR-10 dimensions (3x32x32)
        mock_images = torch.rand(128, 3, 32, 32)
        mock_labels = torch.randint(0, 10, (128,))
        test_set = TensorDataset(mock_images, mock_labels)
        
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def get_cifar100(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Warning: Could not load CIFAR100: {e}. Falling back to random mock dataset.")
        import torch
        from torch.utils.data import TensorDataset
        mock_images = torch.rand(128, 3, 32, 32)
        mock_labels = torch.randint(0, 100, (128,))
        test_set = TensorDataset(mock_images, mock_labels)
        
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def get_imagenet(batch_size=128, data_dir='./data', resize=224):
    transform_list = [transforms.ToTensor()]
    if resize:
        transform_list.insert(0, transforms.Resize(resize))
        transform_list.insert(1, transforms.CenterCrop(resize))
    transform = transforms.Compose(transform_list)
    
    val_dir = os.path.join(data_dir, 'imagenet', 'val')
    if not os.path.exists(val_dir):
        print(f"Warning: ImageNet validation dir not found at {val_dir}. Returning mock ImageNet.")
        import torch
        from torch.utils.data import TensorDataset
        # ImageNet size is typically 3x224x224
        mock_images = torch.rand(128, 3, 224, 224)
        mock_labels = torch.randint(0, 1000, (128,))
        test_set = TensorDataset(mock_images, mock_labels)
    else:
        test_set = datasets.ImageFolder(root=val_dir, transform=transform)
        
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader

