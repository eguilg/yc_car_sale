import tensorflow as tf

from tfTemplate.data_loader.data_generator import DataGenerator
from tfTemplate.models.example_model import ExampleModel
from tfTemplate.trainers.example_trainer import ExampleTrainer
from tfTemplate.utils.config import process_config
from tfTemplate.utils.dirs import create_dirs
from tfTemplate.utils.logger import Logger
from tfTemplate.utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create instance of the model you want
    model = ExampleModel(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
