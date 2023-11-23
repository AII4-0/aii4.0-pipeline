from argparse import ArgumentParser
from copy import deepcopy
import os

from mlem.api import save

from benchmark.data_module import DataModule
from benchmark.model_module import ModelModule


class Train:
    """This class manages the training of a model."""

    def __init__(self, epochs: int) -> None:
        """
        Create an object of `Trainer` class.

        :param epochs: The number of epochs.
        """
        self._epochs = epochs

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("Train")
        parser.add_argument("--epochs", type=int, required=True)
        return parent_parser

    def run(self, model: ModelModule, data: DataModule, output_dir: str) -> None:
        """
        Run a benchmark of a model.

        :param model: The model to benchmark.
        :param data: The data used to benchmark the model.
        :param output_dir: The output directory.
        """

        # Save the initial weights
        weights = deepcopy(model.state_dict())

        # init weight dict
        model_weights_dict = {}

        # Iterate over entities
        for entity, (train_dataloader, test_dataloader) in enumerate(data):
            # Restore the initial weights
            model.load_state_dict(weights)

            # Get the optimizer
            optimizer = model.configure_optimizers()

            # Iterate over epochs
            for epoch in range(self._epochs):

                # Create a list to store the epoch's losses
                losses = []

                # Iterate over train batches
                for batch in train_dataloader:
                    # Perform a train step
                    loss = model.training_step(batch)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Add the current loss to the list
                    losses.append(loss.item())

                # Logging
                print(f"Entity {entity} | Train epoch {epoch} | Loss: {sum(losses) / len(losses)}")

            # add weights to dict
            model_weights_dict[f"entity_{entity}"] = model.state_dict()

        # Save state dictionary using MLEM API
        model_path = "model"
        save(model_weights_dict, model_path, sample_data=None, preprocess=None, postprocess=None)
