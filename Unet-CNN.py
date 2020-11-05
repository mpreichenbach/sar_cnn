# This is an implementation of the model used in Lattari, et al. (2019) - Deep Learning for SAR Image Despeckling
# They used Conv2DTranspose on the upsampling side; may be better to use UpSampling2D
# Originally, no batch normalization. Perhaps add that?

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Conv2DTranspose, Concatenate, Input, Lambda,  MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_loader import eurosat, merced, merced_64
import numpy as np
import matplotlib.pyplot as plt

# Custom loss combining l2 and total variation
def loss(x, y):
    return tf.keras.losses.MSE(x, y) + 0.002 * tf.reduce_mean(tf.image.total_variation(y))

# Wang used learning_rate=0.0002; default is 0.001
opt = Adam(learning_rate=0.001)

# Load synthetic EuroSAT data
# clean, speckle, noisy = eurosat()
# dim = 64

# Load synthetic Merced 256 x 256 data
# clean, speckle, noisy = merced()
# dim = 256

# Load synthetic Merced 64 x 64 data
clean, speckle, noisy = merced_64()
dim = 64
n_images = 67200

clean = np.reshape(clean, (len(clean), dim, dim, 1))
speckle = np.reshape(speckle, (len(speckle), dim, dim, 1))
noisy = np.reshape(noisy, (len(speckle), dim, dim, 1))

clean = clean / 255.
noisy = noisy / 255.

# downsampling path
input_img = Input(shape=(dim, dim, 1))
conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1)
max_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2)
conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(max_1)
conv_4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_3)
max_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)
conv_5 = Conv2D(256, (3, 3), padding='same', activation='relu')(max_2)
conv_6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv_5)
max_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_6)
conv_7 = Conv2D(512, (3, 3), padding='same', activation='relu')(max_3)
conv_8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_7)

# upsampling path
tconv_1 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_8)
concat_1 = Concatenate(axis=-1)([tconv_1, conv_6])
conv_9 = Conv2D(256, (3, 3), padding='same', activation='relu')(concat_1)
conv_10 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv_9)
tconv_2 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_10)
concat_2 = Concatenate(axis=-1)([tconv_2, conv_4])
conv_11 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat_2)
conv_12 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_11)
tconv_3 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_12)
concat_3 = Concatenate(axis=-1)([tconv_3, conv_2])
conv_13 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat_3)
conv_14 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_13)
noise = Conv2D(1, (1, 1), padding='same', activation='softplus')(conv_14)

# division layer that outputs a denoised image
division = Lambda(lambda inputs: inputs[0] / inputs[1])([input_img, noise])
output_img = Activation('sigmoid')(division)

unet_cnn = Model(input_img, output_img)
unet_cnn.compile(optimizer='adam', loss=loss)
unet_cnn.fit(noisy, clean,
             epochs=1,
             batch_size=16,
             shuffle=True,
             validation_split=0.2,
             )

# results
n = 10
examples = np.random.randint(0, n_images, n)

denoised_imgs = unet_cnn.predict(noisy[examples])

plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(clean[examples[i]].reshape(dim, dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(noisy[examples[i]].reshape(dim, dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(denoised_imgs[i].reshape(dim, dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
