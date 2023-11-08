from argparse import ArgumentParser
import json
import os

import torch
from torch import Tensor

from benchmark.data_module import DataModule
from benchmark.model_module import ModelModule
from metrics.metrics import f1_score_from_confusion_matrix, precision_from_confusion_matrix, \
        recall_from_confusion_matrix


class Evaluate:
    """This class manages the evaluation of a model."""
    
    def __init__(self) -> None:
        return

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        return parent_parser

    def run(self, model: ModelModule, data: DataModule, output_dir: str) -> None:
        """
        Run an evaluation of a model.

        :param model: The model to benchmark.
        :param data: The data used to benchmark the model.
        :param output_dir: The output directory.
        """

        # Assuming you have the path to the matrix JSON file
        models_dir = os.path.join(output_dir, "models")
        gl_matrix_file = os.path.join(models_dir, "matrix_{0}_data_{1}.pth".format(model.__class__.__name__, data.dataset.name))

        if not os.path.exists(gl_matrix_file):
            print("Matrix file {0} does not exist".format(gl_matrix_file))
            return

        # Load the global matrix
        gl_matrix = torch.load(gl_matrix_file)

        # Compute global metrics.
        self.compute_global_metrics(gl_matrix, output_dir)

    def compute_global_metrics(self, gl_matrix: dict[str, Tensor], output_dir: str) -> None:
        """
        Compute global metrics.

        :param gl_matrix: The global benchmarked matrix.
        :param output_dir: The output directory.
        """

        # retrieve metrics
        gl_tp, gl_tp_adj, gl_tn, gl_tn_adj, gl_fp, gl_fp_adj, gl_fn, gl_fn_adj = gl_matrix

        # Compute the global precision
        gl_precision = precision_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_precision_adj = precision_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Compute the global recall
        gl_recall = recall_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_recall_adj = recall_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Compute the global F1 scores
        gl_f1_score = f1_score_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_f1_score_adj = f1_score_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Print the global metrics
        self._print_global_metrics({
            "Precision": gl_precision,
            "Precision adjusted": gl_precision_adj,
            "Recall": gl_recall,
            "Recall adjusted": gl_recall_adj,
            "F1 score": gl_f1_score,
            "F1 score adjusted": gl_f1_score_adj
        })

        # Write the global metrics
        self._write_global_metrics({
            "Precision": gl_precision,
            "Precision adjusted": gl_precision_adj,
            "Recall": gl_recall,
            "Recall adjusted": gl_recall_adj,
            "F1 score": gl_f1_score,
            "F1 score adjusted": gl_f1_score_adj
        }, output_dir)

    @staticmethod
    def _print_global_metrics(metrics: dict[str, Tensor]) -> None:
        # Compute the max lengths of the key and the value
        max_len_key = max([len(k) for k in metrics.keys()])
        max_len_value = max([len(str(v)) for v in metrics.values()])

        # Compute the width
        width = 3 * 5 + max_len_key + max_len_value

        # Header
        left_pad = (width - 14) // 2
        print("─" * width)
        print((" " * left_pad) + "Global metrics")
        print("─" * width)

        # Rows
        for k, v in metrics.items():
            v = str(v.numpy())
            left_pad = (max_len_key - len(k)) // 2
            middle_pad = max_len_key - len(k) - left_pad + 5 + (max_len_value - len(v)) // 2
            print("     " + (" " * left_pad) + k + (" " * middle_pad) + v)

        # Footer
        print("─" * width)

    @staticmethod
    def _write_global_metrics(metrics: dict[str, Tensor], output_dir: str) -> None:

        # Create the metrics directory if it does not exist
        output_dir = os.path.join(output_dir, "metrics")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Assuming you have the path to the results file
        eval_file = os.path.join(output_dir, "metrics.json")

        # Transform tensor dict into serialisable items
        json_dict = {k: v.tolist() for k, v in metrics.items()}

        # Write results to json file
        with open(eval_file, 'w') as json_file:
            json.dump(json_dict, json_file)
