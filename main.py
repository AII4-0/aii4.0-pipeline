import os
import sys
from argparse import ArgumentParser

from benchmark.data_module import DataModule
from benchmark.eval_module import Evaluate
from benchmark.test_module import Test
from benchmark.train_module import Train
from benchmark.convert_module import Convert
from benchmark.check_module import Check
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
    parser = Test.add_argparse_args(parser)
    parser = Evaluate.add_argparse_args(parser)
    parser = Convert.add_argparse_args(parser)
    parser = Check.add_argparse_args(parser)

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
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "train",
            "test",
            "evaluate",
            "convert",
            "check",
        ],
        required=False
    )

    # Parse all arguments
    args = parser.parse_args(args=list_of_args)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    # Create the data module
    data_module = create_from_arguments(DataModule, args)

    # Prepare the data
    if args.stage == "prepare":
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
    if args.stage == "train":
        train.run(model, data_module, output_dir)

    # ------------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------------
    # Compute performance metrics
    test = create_from_arguments(Test, args)

    # Benchmark the model
    if args.stage == "test":
        test.run(model, data_module, output_dir)

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------
    # Compute global performance metrics
    evaluate = create_from_arguments(Evaluate, args)

    # Evaluate the model
    if args.stage == "evaluate":
        evaluate.run(model, data_module, output_dir)

    # ------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------
    # Convert model
    convert = create_from_arguments(Convert, args)

    # Convert the model
    if args.stage == "convert":
        convert.run(data_module, output_dir)

    # ------------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------------
    # Check model
    check = create_from_arguments(Check, args)

    # Convert the model
    if args.stage == "check":
        check.run(data_module, output_dir)


if __name__ == "__main__":
    main()
