from argparse import ArgumentParser
from copy import deepcopy
import os

import torch

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

        # Create the output models directory if it does not exist
        output_dir = os.path.join(output_dir, "models")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the initial weights
        weights = deepcopy(model.state_dict())

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

                # Save state dictionary for each entity
                model_file = os.path.join(output_dir, "model_{0}_data_{1}_entity_{2}.pth".format(model.__class__.__name__, data.dataset.name, entity))
                torch.save(model.state_dict(), model_file)