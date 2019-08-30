from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def generator(trainable=True, params={'noise_size': 200, 'strides': (2, 2, 2), 'kernel_size': (4, 4, 4)}):
    """
    Returns a Generator Model
    Args:
        trainable (boolean): Should train the model or not
        params (dict): Dictionary with model parameters
    Returns:
        model (keras.Model): Generator model
    """

    noise_size = params['noise_size']
    strides = params['strides']
    kernel_size = params['kernel_size']

    inputs = Input(shape=(1, 1, 1, noise_size))

    generator_l1 = Deconv3D(filters=512, kernel_size=kernel_size,
                            strides=(1, 1, 1), kernel_initializer='glorot_normal',
                            bias_initializer='zeros', padding='valid')(inputs)
    generator_l1 = BatchNormalization()(generator_l1, training=trainable)
    generator_l1 = Activation(activation='relu')(generator_l1)

    generator_l2 = Deconv3D(filters=256, kernel_size=kernel_size,
                            strides=strides, kernel_initializer='glorot_normal',
                            bias_initializer='zeros', padding='same')(generator_l1)
    generator_l2 = BatchNormalization()(generator_l2, training=trainable)
    generator_l2 = Activation(activation='relu')(generator_l2)

    generator_l3 = Deconv3D(filters=128, kernel_size=kernel_size,
                            strides=strides, kernel_initializer='glorot_normal',
                            bias_initializer='zeros', padding='same')(generator_l2)
    generator_l3 = BatchNormalization()(generator_l3, training=trainable)
    generator_l3 = Activation(activation='relu')(generator_l3)

    generator_l4 = Deconv3D(filters=64, kernel_size=kernel_size,
                            strides=strides, kernel_initializer='glorot_normal',
                            bias_initializer='zeros', padding='same')(generator_l3)
    generator_l4 = BatchNormalization()(generator_l4, training=trainable)
    generator_l4 = Activation(activation='relu')(generator_l4)

    generator_l5 = Deconv3D(filters=1, kernel_size=kernel_size,
                            strides=strides, kernel_initializer='glorot_normal',
                            bias_initializer='zeros', padding='same')(generator_l4)
    generator_l5 = BatchNormalization()(generator_l5, training=trainable)
    generator_l5 = Activation(activation='sigmoid')(generator_l5)

    model = Model(inputs=inputs, outputs=generator_l5)
    print("Generator")
    model.summary()

    return model


def discriminator(trainable=True,
                  params={'input_dim': 64, 'strides': (2, 2, 2), 'kernel_size': (4, 4, 4), 'leak_value': 0.2}):
    """
    Returns a Discriminator Model
    Args:
        trainable (boolean): Train the model or not
        params (dict): Dictionary with model parameters
    Returns:
        model (keras.Model): Discriminator model
    """
    input_dim = params['input_dim']
    strides = params['strides']
    kernel_size = params['kernel_size']
    leak_value = params['leak_value']

    inputs = Input(shape=(input_dim, input_dim, input_dim, 1))

    discriminator_l1 = Conv3D(filters=64, kernel_size=kernel_size,
                              strides=strides, kernel_initializer='glorot_normal',
                              bias_initializer='zeros', padding='same')(inputs)
    discriminator_l1 = BatchNormalization()(discriminator_l1, training=trainable)
    discriminator_l1 = LeakyReLU(leak_value)(discriminator_l1)

    discriminator_l2 = Conv3D(filters=128, kernel_size=kernel_size,
                              strides=strides, kernel_initializer='glorot_normal',
                              bias_initializer='zeros', padding='same')(discriminator_l1)
    discriminator_l2 = BatchNormalization()(discriminator_l2, training=trainable)
    discriminator_l2 = LeakyReLU(leak_value)(discriminator_l2)

    discriminator_l3 = Conv3D(filters=256, kernel_size=kernel_size,
                              strides=strides, kernel_initializer='glorot_normal',
                              bias_initializer='zeros', padding='same')(discriminator_l2)
    discriminator_l3 = BatchNormalization()(discriminator_l3, training=trainable)
    discriminator_l3 = LeakyReLU(leak_value)(discriminator_l3)

    discriminator_l4 = Conv3D(filters=512, kernel_size=kernel_size,
                              strides=strides, kernel_initializer='glorot_normal',
                              bias_initializer='zeros', padding='same')(discriminator_l3)
    discriminator_l4 = BatchNormalization()(discriminator_l4, training=trainable)
    discriminator_l4 = LeakyReLU(leak_value)(discriminator_l4)

    discriminator_l5 = Conv3D(filters=1, kernel_size=kernel_size,
                              strides=(1, 1, 1), kernel_initializer='glorot_normal',
                              bias_initializer='zeros', padding='valid')(discriminator_l4)
    discriminator_l5 = BatchNormalization()(discriminator_l5, training=trainable)
    discriminator_l5 = Activation(activation='sigmoid')(discriminator_l5)

    model = Model(inputs=inputs, outputs=discriminator_l5)
    print("Discriminator")
    model.summary()

    return model
