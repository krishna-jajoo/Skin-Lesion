# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from dataset import SegmentationDataset
from model import UNet
from utils import dice_loss

# Set paths
image_dir = "./data/images/Dataset 1"
mask_dir = "./data/images/Dataset 1"

image_paths = sorted(
    [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".bmp")
    ]
)
mask_paths = sorted(
    [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
)

X_train, X_val, y_train, y_val = train_test_split(
    image_paths, mask_paths, test_size=0.1, random_state=42
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = SegmentationDataset(X_train, y_train, transform=transform)
val_dataset = SegmentationDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")

# Save model
os.makedirs("training/outputs", exist_ok=True)
# torch.save(model.state_dict(), "training/outputs/unet_segmentation_model.pth")
model.save("application/outputs/unet_segmentation_model.h5")
