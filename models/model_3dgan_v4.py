from keras.layers import Dense, Reshape, UpSampling3D, Flatten, LeakyReLU, ZeroPadding3D, Input, Embedding, Multiply, \
    Dropout, AveragePooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model


def generator():
    """
    Returns a Generator Model
    Returns:
        model (keras.Model): Generator model
    """

    # filters = 64
    # kernel_size = (4, 4, 4)
    # strides = (1, 1, 1)
    #
    # model = Sequential()
    # model.add(Dense(filters * 16 * 16 * 16), input_dim=100)
    # model.add(Reshape((16, 16, 16, filters)))
    # model.add(Conv3D(filters, kernel_size, strides=strides, padding='same'))

    # model.add(Activation('tanh'))
    # model.add(Dense(filters * 16 * 16 * 16))
    # model.add(BatchNormalization())
    # model.add(Activation('tanh'))
    # model.add(Reshape((16, 16, 16, filters), input_shape=(filters * 16 * 16 * 16,)))
    # model.add(UpSampling3D(size=(2, 2, 2)))
    # model.add(Conv3D(filters, kernel_size, strides=strides, padding='same'))
    # model.add(Activation('tanh'))
    # model.add(UpSampling3D(size=(2, 2, 2)))
    # model.add(Conv3D(1, kernel_size, strides=strides, padding='same'))
    # model.add(Activation('tanh'))


def discriminator():
    """
    Returns a Discriminator Model
    Returns:
        model (keras.Model): Discriminator model
    """

    image = Input(shape=(64, 64, 64, 1))

    x = Conv3D(32, 5, 5, 5, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)

    print(Model(image, h).summary())

    image = Input(shape=(64, 64, 64, 1))

    dnn_out = dnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='sigmoid', name='auxiliary')(dnn_out)

    # filters = 64
    # kernel_size = (4, 4, 4)
    # strides = (1, 1, 1)
    #
    # model = Sequential()
    # model.add(
    #     Conv3D(filters, kernel_size, strides=strides,
    #            padding='same',
    #            input_shape=(64, 64, 64, 1))
    # )
    # model.add(Activation('tanh'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Conv3D(filters, kernel_size, strides=strides))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    print("Discriminator V3")
    print(Model(input=image, output=[fake, aux]).summary())

    return Model(input=image, output=[fake, aux])
