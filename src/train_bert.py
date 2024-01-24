#!/usr/bin/env python
# coding: utf-8

import argparse
from utils.preprocess_utils import preprocess_text
from utils.pipeline_utils import run_cv
from utils.data_utils import read_training_data
from models.bert_model import BertModel
from sklearn.model_selection import train_test_split
import time
from utils.constants import (
    MODEL_CV_RESULT_PATH,
    TARGET_INV_DICT,
    TARGET_INV_DICT_FASHION,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from utils.preprocess_utils import special_token
import pandas as pd
import matplotlib.pyplot as plt


def main(args):
    df = read_training_data(args.data_path)

    bias_naming = ""
    if args.prevent_bias == 2:
        bias_naming = "-fully-unbiased"
    elif args.prevent_bias == 1:
        bias_naming = "-casing-unbiased"

    experiment_name = f"text_classification_v2-{args.name}"

    model_params = {
        "model_path": args.model_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tokenizer_max_len": args.tokenizer_max_len,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "llrd_decay": args.llrd_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "prevent_bias": args.prevent_bias,
        "mlm_pretrain": args.mlm_pretrain,
        "mlm_probability": args.mlm_probability,
        "out_folder": args.out_folder,
        "experiment_name": experiment_name,
        "num_labels": args.num_labels,
    }
    if args.cv:
        run_cv(
            model_obj=BertModel,
            model_params=model_params,
            input_df=df,
            fold_col=args.fold_name,
            x_col=args.xcol,
            y_col=args.ycol,
            experiment_name=experiment_name,
            add_to_zoo=args.add_zoo,
            is_nn=True,
            prevent_bias=args.prevent_bias,
        )
    else:
        print()
        print("*" * 30)
        print("Started Training")
        print("*" * 30)
        print(f"Experiment: '{experiment_name}'")
        print("*" * 30)
        print()
        start_time = time.time()

        X = df[args.xcol]
        y = df[args.ycol]

        # Split the data into 90% train and 10% validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        model = BertModel(**model_params)
        model.train(X_train, y_train, X_val, y_val)
        preds, pred_probas = model.predict(X_val)

        if args.fashion:
            target_inv_dict = TARGET_INV_DICT_FASHION
        else:
            target_inv_dict = TARGET_INV_DICT

        preds = [target_inv_dict[p] for p in preds]
        y_val = [target_inv_dict[p] for p in y_val]
        # Classification Report
        print()
        print("*" * 30)
        print("Classification Report:")
        print(classification_report(y_val, preds))
        print("*" * 30)

        # Confusion Matrix
        print()
        print("*" * 30)
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_val, preds)
        print(conf_matrix)

        # Confusion Matrix Display
        class_labels = list(target_inv_dict.values())
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=class_labels
        )
        # Get the figure and axis
        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot the confusion matrix on the specified axis
        disp.plot(ax=ax)
        plt.savefig(
            f"{args.out_folder}/{experiment_name}/confusion_matrix_validation.png"
        )

        y_val_column = "related_product" if args.fashion else "category"
        x_val_column = "title" if args.fashion else "description_text"
        # Create a DataFrame for wrong predictions
        wrong_predictions = pd.DataFrame(
            {
                "id": df.loc[X_val.index, "id"],  # Assuming "id" is the column name
                x_val_column: X_val,
                y_val_column: y_val,
                "predicted": preds,
            }
        )

        probas_list = pred_probas
        predicted_probs = [
            sorted(
                zip(range(args.num_labels), probas), key=lambda x: x[1], reverse=True
            )
            for probas in probas_list
        ]
        wrong_predictions["probas"] = predicted_probs

        # Filter for wrong predictions
        wrong_predictions = wrong_predictions[
            wrong_predictions[y_val_column] != wrong_predictions["predicted"]
        ]

        # Save wrong predictions to a CSV file
        wrong_predictions.to_csv(
            f"{args.out_folder}/{experiment_name}/validation_wrongs.csv", index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, default="dbmdz/bert-base-turkish-128k-uncased"
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tokenizer_max_len", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--fashion", type=bool, default=False, help="Is it category prediction"
    )
    parser.add_argument("--num_labels", type=int, default=3, help="number of labels")

    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--llrd_decay", type=float, default=0.95)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--prevent_bias", type=int, default=0)

    parser.add_argument("--mlm_pretrain", action="store_true")
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--out_folder", type=str, default="../checkpoint")
    parser.add_argument("--fold_name", type=str, default="public_fold")
    parser.add_argument("--xcol", type=str, default="text")
    parser.add_argument("--ycol", type=str, default="target")
    parser.add_argument("--add_zoo", action="store_true")
    parser.add_argument("--cv", action="store_true")
    args = parser.parse_args()
    main(args)
