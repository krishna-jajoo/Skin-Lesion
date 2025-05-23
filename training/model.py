from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    ReLU,
    Concatenate,
)
from tensorflow.keras import models  # layers,


def encoder_block(input, num_filters):
    conv_b = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(conv_b)
    return conv_b, p


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def decoder_block(input, skip_feature, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_feature])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_size=(128, 128, 3)):
    # inputs = Input((128, 128, 3))
    # inputs = Input(input_size)
    inputs = Input(input_size)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(d4)

    return models.Model(inputs, outputs)


# def build_unet(input_size=(128, 128, 3)):
#     inputs = layers.Input(input_size)

#     # Encoder
#     c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
#     c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)

#     c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
#     c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)

#     c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
#     c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(c3)
#     p3 = layers.MaxPooling2D((2, 2))(c3)

#     c4 = layers.Conv2D(512, 3, activation="relu", padding="same")(p3)
#     c4 = layers.Conv2D(512, 3, activation="relu", padding="same")(c4)
#     p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

#     # Bottleneck
#     c5 = layers.Conv2D(1024, 3, activation="relu", padding="same")(p4)
#     c5 = layers.Conv2D(1024, 3, activation="relu", padding="same")(c5)

#     # Decoder
#     u6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(c5)
#     u6 = layers.concatenate([u6, c4])
#     c6 = layers.Conv2D(512, 3, activation="relu", padding="same")(u6)
#     c6 = layers.Conv2D(512, 3, activation="relu", padding="same")(c6)

#     u7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(c6)
#     u7 = layers.concatenate([u7, c3])
#     c7 = layers.Conv2D(256, 3, activation="relu", padding="same")(u7)
#     c7 = layers.Conv2D(256, 3, activation="relu", padding="same")(c7)

#     u8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c7)
#     u8 = layers.concatenate([u8, c2])
#     c8 = layers.Conv2D(128, 3, activation="relu", padding="same")(u8)
#     c8 = layers.Conv2D(128, 3, activation="relu", padding="same")(c8)

#     u9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c8)
#     u9 = layers.concatenate([u9, c1])
#     c9 = layers.Conv2D(64, 3, activation="relu", padding="same")(u9)
#     c9 = layers.Conv2D(64, 3, activation="relu", padding="same")(c9)

#     outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

#     return models.Model(inputs, outputs)
