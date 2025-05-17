import tensorflow as tf

# import os
from PIL import Image
import numpy as np


class SegmentationDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=8, image_size=(128, 128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_x, batch_y):
            img = Image.open(img_path).convert("RGB").resize(self.image_size)
            mask = Image.open(mask_path).convert("L").resize(self.image_size)

            img = np.array(img) / 255.0
            mask = np.array(mask) / 255.0
            mask = np.expand_dims((mask > 0).astype(np.float32), axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)
