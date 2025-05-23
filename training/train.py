import os
# import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from training.dataset import SegmentationDataset
from training.model import build_unet
from training.utils import dice_loss

# Paths
image_dir = "./data/images/Dataset 1"
mask_dir = "./data/images/Dataset 1"

image_paths = sorted(
    [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".bmp")]
)
mask_paths = sorted(
    [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
)

X_train, X_val, y_train, y_val = train_test_split(
    image_paths, mask_paths, test_size=0.1, random_state=42
)

train_dataset = SegmentationDataset(X_train, y_train, batch_size=8)
val_dataset = SegmentationDataset(X_val, y_val, batch_size=8)

# Model
model = build_unet()
model.compile(optimizer=Adam(1e-3), loss=dice_loss, metrics=["accuracy"])

# Callbacks
os.makedirs("application/outputs", exist_ok=True)
checkpoint = ModelCheckpoint(
    "application/outputs/unet_segmentation_model.h5",
    save_best_only=True,
    monitor="val_loss",
)

# Training
model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[checkpoint])
