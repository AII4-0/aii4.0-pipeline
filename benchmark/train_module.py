from argparse import ArgumentParser
from copy import deepcopy
import os

from mlem.api import save
import torch
import yaml

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

        # Load the initial weights
        model.load_state_dict(weights)

        # Iterate over entities
        for entity, (train_dataloader, test_dataloader) in enumerate(data):
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

            # Save PyTorch model
            export_model_dir = os.path.join(output_dir, "model")
            os.makedirs(export_model_dir, exist_ok=True)
            torch.save(model, os.path.join(export_model_dir, f"model_{entity}.pth"))

        # Save model using MLEM API
        save(model, "model", sample_data=None, preprocess=None, postprocess=None)
