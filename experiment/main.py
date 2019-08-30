import datetime
from trainer import Trainer

"""
Main
"""


def main():
    # data = DataSet(batch_size=32)
    # data.get_batch_volumes()
    experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = "GAN3D"
    batch_size = 30
    nb_epoch = 10000
    input_dim = 64

    # Initiate Trainer
    trainer = Trainer(experiment_id, model_name, batch_size, nb_epoch, input_dim)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
