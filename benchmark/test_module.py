from argparse import ArgumentParser
import os
from pathlib import Path

from mlem.api import load
import torch

from benchmark.data_module import DataModule
from benchmark.model_module import ModelModule
from metrics.metrics import confusion_matrix, f1_score_from_confusion_matrix, precision_from_confusion_matrix, \
    recall_from_confusion_matrix
from metrics.point_adjustment import adjust_pred
from metrics.thresholding import best_threshold


class Test:
    """This class manages the testing of a model."""

    def __init__(self, export_folder: Path = None) -> None:
        """
        Create an object of `Trainer` class.
        :param export_folder: The path to the folder where the model weights should be saved.
        """
        self._export_folder = export_folder

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
        Run a benchmark of a model.

        :param model: The model to benchmark.
        :param data: The data used to benchmark the model.
        :param output_dir: The input model directory.
        """

        # Retrieve model using MLEM
        if self._export_folder:
            model = load(self._export_folder)

        # Initialize global confusion matrix variables
        gl_tp = torch.tensor(0)
        gl_tp_adj = torch.tensor(0)
        gl_tn = torch.tensor(0)
        gl_tn_adj = torch.tensor(0)
        gl_fp = torch.tensor(0)
        gl_fp_adj = torch.tensor(0)
        gl_fn = torch.tensor(0)
        gl_fn_adj = torch.tensor(0)

        # Iterate over entities
        for entity, (train_dataloader, test_dataloader) in enumerate(data):

            # Set the model to eval mode
            model.eval()

            # Disable gradient calculation
            with torch.no_grad():

                # Create lists to store batch results
                y_pred_list = []
                y_true_list = []

                # Iterate over test batches
                for batch in test_dataloader:
                    # Perform the prediction
                    pred, true = model.test_step(batch)

                    # Add results to the lists
                    y_pred_list.append(pred)
                    y_true_list.append(true)

                # Convert lists to tensors
                y_pred = torch.cat(y_pred_list)
                y_true = torch.cat(y_true_list)

                # Find the best thresholds
                threshold = best_threshold(y_true, y_pred)
                threshold_adj = best_threshold(y_true, y_pred, point_adjustment=True)

                # Binarize the predictions with the corresponding threshold
                y_pred_std = torch.where(y_pred >= threshold, 1, 0)
                y_pred_adj = torch.where(y_pred >= threshold_adj, 1, 0)

                # Apply the point adjustment to the prediction
                y_pred_adj = adjust_pred(y_true, y_pred_adj)

                # Compute the confusion matrices
                tp, tn, fp, fn = confusion_matrix(y_true, y_pred_std)
                tp_adj, tn_adj, fp_adj, fn_adj = confusion_matrix(y_true, y_pred_adj)

                # Compute the precision
                precision = precision_from_confusion_matrix(tp, tn, fp, fn)
                precision_adj = precision_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

                # Logging
                print(f"Entity {entity} | Test          | Precision: {precision}, Precision adjusted: {precision_adj}")

                # Compute the recall
                recall = recall_from_confusion_matrix(tp, tn, fp, fn)
                recall_adj = recall_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

                print(f"Entity {entity} | Test          | Recall: {recall}, Recall adjusted: {recall_adj}")

                # Compute the F1 scores
                f1_score = f1_score_from_confusion_matrix(tp, tn, fp, fn)
                f1_score_adj = f1_score_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

                print(f"Entity {entity} | Test          | F1: {f1_score}, F1 adjusted: {f1_score_adj}")

                # Update the global confusion matrix
                gl_tp += tp
                gl_tp_adj += tp_adj
                gl_tn += tn
                gl_tn_adj += tn_adj
                gl_fp += fp
                gl_fp_adj += fp_adj
                gl_fn += fn
                gl_fn_adj += fn_adj

                # Set the model to train mode
                model.train()

        # Create the output directory if it does not exist
        output_dir = os.path.join(output_dir, "confusion")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gl_matrix = [gl_tp, gl_tp_adj, gl_tn, gl_tn_adj, gl_fp, gl_fp_adj, gl_fn, gl_fn_adj]
        gl_matrix_file = os.path.join(output_dir,
                                      "matrix_{0}_data_{1}.pth".format(model.__class__.__name__, data.dataset.name))

        # Save global confusion matrix
        torch.save(gl_matrix, gl_matrix_file)
