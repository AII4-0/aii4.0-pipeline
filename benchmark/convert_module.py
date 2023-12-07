from argparse import ArgumentParser
import os
from pathlib import Path

import nobuco
import tensorflow as tf
import torch
from nobuco import ChannelOrder
import numpy as np

from benchmark.data_module import DataModule

test_dataloader = None


class Convert:
    """This class manages the export of a model."""

    def __init__(self, entity: int, export_model: Path = None) -> None:
        """
        Create an object of `Trainer` class.
        :param entity: The entity to convert.
        :param export_model: The path to the folder where the exported model is saved.
        """
        self._entity = entity
        self._export_model = export_model

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("Convert")
        parser.add_argument("--entity", type=int, required=False)
        return parent_parser

    def representative_data_gen(self):
        global test_dataloader
        iterator = iter(test_dataloader)
        length = len(iterator)

        for i in range(length):
            # Model has only one input so each data point has one element.
            # data = np.random.rand(1, 99, 1)
            data = next(iterator)
            data = data[0][:, : -1]
            data = data.numpy()

            yield [data.astype(np.float32)]

    def convert_to_c(self, tflite_model, file_name, path0):
        from tensorflow.lite.python.util import convert_bytes_to_c_source
        source_text, header_text = convert_bytes_to_c_source(tflite_model, file_name)

        if not os.path.exists(path0):
            os.makedirs(path0)

        with open(os.path.join(path0, file_name + '.h'), 'w') as file:
            file.write(header_text)

        with open(os.path.join(path0, file_name + '.cpp'), 'w') as file:
            file.write("\n#include \"" + file_name + ".h\"\n")
            file.write(source_text)

    def run(self, data: DataModule, output_dir: str) -> None:
        """
        Export a model.

        :param data: The data used to benchmark the model.
        :param output_dir: The exported model directory.
        """

        # ensure the pytorch model has been created
        if self._export_model is None:
            return

        # Get the train dataloader for the entity
        global test_dataloader

        _, test_dataloader = data[self._entity]

        # Get a data sample
        data_sample = next(iter(test_dataloader))
        x = data_sample[0][:, : -1]

        # Load the PyTorch model
        pytorch_model = torch.load(os.path.join(output_dir, "model", f"model_{self._entity}.pth"))

        # Disable gradient calculation to test the model
        torch.autograd.set_grad_enabled(False)
        #output = torch.tensor([[[0.0]]])
        out = pytorch_model(x)

        # Convert the PyTorch model to Keras model
        keras_model = nobuco.pytorch_to_keras(
            pytorch_model,
            args=[x],
            inputs_channel_order=ChannelOrder.PYTORCH,
            outputs_channel_order=ChannelOrder.PYTORCH
        )

        ############################################
        # Convert the Keras model to TensorFlow lite
        ############################################
        print("Convert and save tf lite model")
        tflite_model = tf.lite.TFLiteConverter.from_keras_model(keras_model).convert()

        # Save the model
        with open(self._export_model.parent.joinpath(self._export_model.stem + ".tflite"), "wb") as f:
            f.write(tflite_model)

        # convert to C source code and store it
        print("convert to C source code")
        self.convert_to_c(tflite_model, self._export_model.stem, os.path.join(output_dir, "export"))

        ############################################
        # Convert keras model into quantized tf lite model
        ############################################
        print("Convert and save tf lite model - quantized")
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_data_gen
        # converter.target_spec.supported_types = [tf.float16]
        # converter.exclude_conversion_metadata = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Convert the Keras model to TensorFlow lite quant
        tflite_model_quant = converter.convert()

        # Save the model
        with open(self._export_model.parent.joinpath(self._export_model.stem + "_quant.tflite"), "wb") as f:
            f.write(tflite_model_quant)

        # convert to C source code and store it
        self.convert_to_c(tflite_model_quant, self._export_model.stem + "_quant", os.path.join(output_dir, "export"))
