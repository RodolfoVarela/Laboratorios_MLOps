# src/data_loader.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

imagenet_stats = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

def get_dataloaders(root_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Crea un DataLoader de ImageFolder sobre root_dir y devuelve:
      loader, class_names
    root_dir debe tener subcarpetas por clase con imágenes dentro.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_stats["mean"], std=imagenet_stats["std"]),
    ])
    ds = datasets.ImageFolder(root_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, ds.classes
