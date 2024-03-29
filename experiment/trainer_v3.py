import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib

from datautils.data import DataSet
from datautils.utils import plot_volumes_as_images
from models.model_3dgan_v3 import discriminator as discriminator_model
from models.model_3dgan_v3 import generator as generator_model


class Trainer():
    """
    Trainer Class
    """

    def __init__(self, experiment_id, model_name, batch_size, nb_epoch, input_dim):
        """
        Initialize the Trainer class
        Args:
            experiment_id: Experiment ID
            model_name: Name of the model
            batch_size: Batch Size
            nb_epoch: Number of epoch
            input_dim: Input dimension
        """
        self.nb_epoch = nb_epoch
        self.model_name = model_name
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.beta = 0.5
        self.noise_size = 100
        self.noise_loc = -1
        self.noise_scale = 1
        self.image_save_interval = 100
        self.weight_save_interval = 1000

        self.data = DataSet(batch_size=self.batch_size)

        # GPU Stuff
        local_device_protos = device_lib.list_local_devices()
        list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        self.num_gpus = len(list_of_gpus)

        print("Trainer init finished")

    def adversarial_model(self, generator, discriminator):
        """
        Creates a adversarial model: which is a generator and discriminator stacked together
        :return: adversarial model
        """
        # optimizer = RMSprop(lr=0.0001, decay=3e-8)
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model

    def train(self):
        """
        Train function does the training manually
        :return: None
        """
        # Get the generator and discriminator models
        generator = generator_model()
        discriminator = discriminator_model()

        # Get the adversarial model
        adversarial = self.adversarial_model(generator, discriminator)

        generator_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)

        generator = multi_gpu_model(generator, gpus=self.num_gpus)
        generator.compile(loss='binary_crossentropy', optimizer="SGD", metrics=['accuracy'])

        discriminator_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)

        adversarial = multi_gpu_model(adversarial, gpus=self.num_gpus)
        adversarial.compile(loss='binary_crossentropy', optimizer=generator_optimizer, metrics=['accuracy'])
        discriminator.trainable = True

        discriminator = multi_gpu_model(discriminator, gpus=self.num_gpus)
        discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

        # Create Sample noise
        noise_sample = np.random.uniform(self.noise_loc, self.noise_scale, size=(self.batch_size, self.noise_size))

        print(noise_sample.shape)

        # Run epochs manually
        for epoch in range(self.nb_epoch):
            # print ("Epoch is: " + str(epoch))

            # Get volumes batch
            train_volumes = self.data.get_batch_volumes()
            # print("X: " + str(train_volumes.shape))

            # Create random noise
            noise = np.random.uniform(self.noise_loc, self.noise_scale, size=(self.batch_size, self.noise_size))
            # print("noise: " + str(noise.shape))

            # Predict noise volumes
            fake_volumes = generator.predict(noise, verbose=0)
            # print("generated_volumes: " + str(fake_volumes.shape))

            X = np.concatenate((train_volumes, fake_volumes))
            # print("X: " + str(X.shape))
            # print([0])
            Y = np.reshape([1] * self.batch_size + [0] * self.batch_size, (-1, 1))
            # print("Y: " + str(Y.shape))

            # We give the data to discriminator for training

            discriminator_loss = discriminator.train_on_batch(X, Y)
            # print("Epoch is: " + str(epoch) + "    Discriminator_loss : " + str(discriminator_loss))

            # Create random noise
            noise = np.random.uniform(self.noise_loc, self.noise_scale, size=(self.batch_size, self.noise_size))
            # Train the adversarial model
            Y = np.reshape([1] * self.batch_size, (-1, 1))
            # print("Y: " + str(Y.shape))
            discriminator.trainable = False
            adversarial_loss = adversarial.train_on_batch(noise, Y)
            discriminator.trainable = True
            log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, discriminator_loss[0], discriminator_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, adversarial_loss[0], adversarial_loss[1])
            print(log_mesg)

            if epoch % self.weight_save_interval == 10:
                generator.save_weights(
                    '/DeployedProjects/simple-3d-gan/data/weights/generator_' + str(
                        epoch), True)
                discriminator.save_weights(
                    'DeployedProjects/simple-3d-gan/data/weights/discriminator_' + str(
                        epoch), True)

            if epoch % self.image_save_interval == 10:
                generated_volumes = generator.predict(noise_sample, verbose=0)
                # save_volumes_as_images(generated_volumes, n=10, prefix=str(epoch))
                plot_volumes_as_images(generated_volumes)
