import os
import sys
from argparse import ArgumentParser

from benchmark.data_module import DataModule
from benchmark.eval_module import Evaluate
from benchmark.test_module import Test
from benchmark.train_module import Train
from models.gan import GAN
from models.lstm import LSTM
from models.tran_ad import TranAD
from models.transformer import Transformer
from models.vae import VAE
from utils.arguments import augment_arguments_with_yaml, namespace_to_list, create_from_arguments


def main() -> None:
    """Execute the whole program."""
    # ------------------------------------------------------------------------
    # Manage args
    # ------------------------------------------------------------------------
    # Create the argument parser
    parser = ArgumentParser()

    # Arguments from YAML configuration file ?
    parser.add_argument("--config", type=str)

    # Parse the known arguments
    args, _ = parser.parse_known_args()

    # If the arguments are stored in YAML configuration file, load them
    if args.config:
        args = augment_arguments_with_yaml(args, args.config)

    # Add benchmark arguments
    parser = Train.add_argparse_args(parser)

    # Which model?
    parser.add_argument(
        "--model",
        choices=[
            "LSTM",
            "Transformer",
            "VAE",
            "GAN",
            "TranAD"
        ],
        required=True
    )

    # Add data module arguments
    parser = DataModule.add_argparse_args(parser)

    # Create a list of arguments
    list_of_args = namespace_to_list(args) + sys.argv[1:]

    # Parse the known arguments
    args, _ = parser.parse_known_args(args=list_of_args)

    # Get the model class
    if args.model == "LSTM":
        model_cls = LSTM
    elif args.model == "Transformer":
        model_cls = Transformer
    elif args.model == "VAE":
        model_cls = VAE
    elif args.model == "GAN":
        model_cls = GAN
    elif args.model == "TranAD":
        model_cls = TranAD
    else:
        raise RuntimeError("Unrecognized model")

    # Add model specific arguments
    parser = model_cls.add_argparse_args(parser)

    # Add pipeline step arguments
    parser.add_argument("--prepare", action="store_true", help='Execute preparation step')
    parser.add_argument("--train", action="store_true", help='Execute training step')
    parser.add_argument("--test", action="store_true", help='Execute testing step')
    parser.add_argument("--evaluate", action="store_true", help='Execute evaluation step')

    # Parse all arguments
    args = parser.parse_args(args=list_of_args)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    # Create the data module
    data_module = create_from_arguments(DataModule, args)

    # Prepare the data
    if args.prepare:
        data_module.run()

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------
    model = create_from_arguments(model_cls, args, in_channels=data_module.dataset.dimension)

    # model output directory
    output_folder = "output"
    output_dir = os.path.join('.', output_folder)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------
    # Create the benchmark
    train = create_from_arguments(Train, args)

    # Train the model
    if args.train:
        train.run(model, data_module, output_dir)

    # ------------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------------
    # Compute performance metrics
    test = create_from_arguments(Test, args)

    # Benchmark the model
    if args.test:
        test.run(model, data_module, output_dir)

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------
    # Compute global performance metrics
    evaluate = create_from_arguments(Evaluate, args)

    # Evaluate the model
    if args.evaluate:
        evaluate.run(model, data_module, output_dir)


if __name__ == "__main__":
    main()
