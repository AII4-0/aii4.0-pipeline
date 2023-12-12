from argparse import ArgumentParser
import json
import os

import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import mean_squared_error

from benchmark.data_module import DataModule


class Check:
    """This class manages the export of a model."""

    def __init__(self, entity: int) -> None:
        """
        Create an object of `Trainer` class.
        :param entity: The entity to convert.
        Whether
        """
        self._entity = entity

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        return parent_parser
    
    def run(self, data: DataModule, output_dir: str) -> None:
        """
        Run a benchmark of a model.

        :param data: The data used to benchmark the model.
        :param output_dir: The output directory.
        """

        # Get the train dataloader for the entity
        global test_dataloader

        y, test_dataloader = data[self._entity]
        
        # Disable gradient calculation
        torch.autograd.set_grad_enabled(False)
    
        # Load the PyTorch model
        pytorch_model = torch.load(os.path.join(output_dir, "model", f"model_{self._entity}.pth"))
        pytorch_model.eval()
    
        # Load the TFLite model and allocate tensors
        tflite_model = "model.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
        interpreter.allocate_tensors()
    
        # Get the TFLite input/output tensor index
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
    
        # Create lists to store the predictions
        pytorch_predictions = []
        tflite_predictions = []
    
        # Iterate over test batches
        i_input = 0
        for x, y in test_dataloader:
            # Perform the prediction with the PyTorch model
            x = x[:, : -1]
            # y = torch.tensor([[[0.0]]])
            # pytorch_pred = pytorch_model(x, y)
            pytorch_pred = pytorch_model(x)
    
            # Perform the prediction with the TFLite model
            interpreter.set_tensor(input_index, x.numpy())
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_index)
    
            # Add predictions to the lists
            pytorch_predictions.append(pytorch_pred.numpy().squeeze())
            tflite_predictions.append(tflite_pred.squeeze())
    
            # Print input data
            # if i_input >=400 and i_input < 430:
            #     print(f'Input i: {i_input} data : {x.numpy().reshape(-1)}')
    
            i_input = i_input + 1
    
        # Convert lists to tensors
        pytorch_predictions = np.array(pytorch_predictions)
        tflite_predictions = np.array(tflite_predictions)
    
        # Compute the mean squared error
        mae = mean_squared_error(pytorch_predictions.flatten(), tflite_predictions.flatten())

        # Logging
        print(f"Mean squared error: {mae:.20f}")

        # Write the check metrics
        self._write_check_metrics({
            "Mean squared error": mae,
            "Pytorch prediction": pytorch_predictions,
            "Tflite prediction": tflite_predictions,
            "Tflite prediction[0:100]": tflite_predictions[0:100],
            "Tflite prediction[400:500]": tflite_predictions[400:500]
        }, output_dir)

    @staticmethod
    def _write_check_metrics(metrics: dict[str, any], output_dir: str) -> None:

        # Create the metrics directory if it does not exist
        output_dir = os.path.join(output_dir, "metrics")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Assuming you have the path to the results file
        eval_file = os.path.join(output_dir, "check.json")

        # Transform tensor dict into serialisable items
        json_dict = {k: v.tolist() for k, v in metrics.items()}

        # Write results to json file
        with open(eval_file, 'w') as json_file:
            json.dump(json_dict, json_file)
