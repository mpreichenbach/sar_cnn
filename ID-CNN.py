# this is an implementation of the model in Wang, et al. - SAR Image Despeckling Using a Convolutional Neural Network

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_loader import eurosat, merced, merced_64
import numpy as np
import matplotlib.pyplot as plt

# factor for the custom loss function; Wang used factor = 0.002
factor = 0.002
# Custom loss combining l2 and total variation


def loss(y_true, y_pred):
    loss_l1 = tf.reduce_sum(tf.image.total_variation(y_pred))
    loss_l2 = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss_l2 + 0.002 * loss_l1

tf.keras.losses.loss = loss
print(tf.keras.losses.loss)

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

# number of filters in each layer (Wang used 64)
filters = 64
# specifies the layers of the model (note that images in Wang were 256x256.
input_img = Input(shape=(dim, dim, 1))
# L1
x = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_img)
# L2
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L3
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L4
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L5
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L6
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L7
x = Conv2D(filters, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# L8 - Generates the noise signature
noise = Conv2D(1, (3, 3), padding='same', activation='softplus')(x)
# L9 is the "division layer"
division = Lambda(lambda inputs: inputs[0] / inputs[1])([input_img, noise])
# L10 yields an image with pixels in [0, 1]; Wang used tanh here, but maybe use sigmoid?
decoded = Activation('relu')(division)

id_cnn = Model(input_img, decoded)

# here we build the model with the custom loss function
id_cnn.compile(optimizer=opt, loss=loss)
id_cnn.fit(noisy, clean,
           epochs=1,
           batch_size=16,
           shuffle=True,
           validation_split=0.2,
           )

n = 10
examples = np.random.randint(0, n_images, n)

denoised_imgs = id_cnn.predict(noisy[examples])

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
