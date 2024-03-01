import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, precision_recall_curve
from tqdm import tqdm
from torch.nn.functional import softmax
import argparse
import json
import os
import logging
from utils.constants import TARGET_INV_DICT, TARGET_DICT
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define relu_evidence function
def relu_evidence(x):
    return torch.relu(x)

def load_config(config_path):
    with open(config_path, 'r') as json_file:
        config_data = json.load(json_file)
    return config_data

def initialize_tokenizer(tokenizer_name):
    return BertTokenizer.from_pretrained(tokenizer_name)

def load_model(model_path, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()
    return model

def prepare_test_data(test_path, tokenizer):
    test_df = pd.read_csv(test_path)
    input_texts = [str(text) for text in test_df['description_text'].tolist()]
    input_ids = tokenizer.batch_encode_plus(input_texts, add_special_tokens=True, truncation=True, padding='max_length', return_tensors='pt')
    return test_df, input_ids

class TestDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids['input_ids']
        self.attention_mask = input_ids['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

def evaluate_model(model, test_loader, num_labels, device, use_uncertainty=False):
    all_probas = []
    all_predictions = []
    uncertainties = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            for output in outputs.logits:
                if use_uncertainty:
                    evidence = relu_evidence(output)
                    alpha = evidence + 1
                    uncertainty = num_labels / torch.sum(alpha, dim=-1, keepdim=True)
                    prob = alpha / torch.sum(alpha, dim=-1, keepdim=True)
                    uncertainties.append(uncertainty)
                else:
                    prob = softmax(output, dim=-1)
                _, preds = torch.max(output, -1)
                all_probas.append(prob.tolist())
                all_predictions.append(preds.tolist())
    if use_uncertainty:
        return all_probas, all_predictions, uncertainties
    else:
        return all_probas, all_predictions

def calculate_precision_at_thresholds(targets, probas, thresholds):
    probas_array = np.array(probas)  # Convert probas to a NumPy array
    class_precisions = []
    num_classes = len(np.unique(targets))
    
    # Convert targets to a NumPy array for comparison
    targets_array = np.array(targets)
    
    for class_idx in range(num_classes):
        class_probs = probas_array[:, class_idx]
        class_precision = []
        
        for threshold in thresholds:
            predictions = (class_probs >= threshold).astype(int)
            # Convert boolean array to integer array
            class_targets = (targets_array == class_idx).astype(int)
            precision = precision_score(class_targets, predictions, average='binary', zero_division=0)
            class_precision.append(precision)
        
        class_precisions.append(class_precision)
    
    return class_precisions

def calculate_recall_at_thresholds(targets, probas, thresholds):
    probas_array = np.array(probas)  # Convert probas to a NumPy array
    class_recalls = []
    num_classes = len(np.unique(targets))
    
    # Convert targets to a NumPy array for comparison
    targets_array = np.array(targets)
    
    for class_idx in range(num_classes):
        class_probs = probas_array[:, class_idx]
        class_recall = []
        
        for threshold in thresholds:
            predictions = (class_probs >= threshold).astype(int)
            # Convert boolean array to integer array
            class_targets = (targets_array == class_idx).astype(int)
            recall = recall_score(class_targets, predictions, average='binary', zero_division=0)
            class_recall.append(recall)
        
        class_recalls.append(class_recall)
    
    return class_recalls

def calculate_f1_at_thresholds(targets, probas, thresholds):
    probas_array = np.array(probas)  # Convert probas to a NumPy array
    class_f1_scores = []
    num_classes = len(np.unique(targets))
    
    # Convert targets to a NumPy array for comparison
    targets_array = np.array(targets)
    
    for class_idx in range(num_classes):
        class_probs = probas_array[:, class_idx]
        class_f1_score = []
        
        for threshold in thresholds:
            predictions = (class_probs >= threshold).astype(int)
            # Convert boolean array to integer array
            class_targets = (targets_array == class_idx).astype(int)
            f1 = f1_score(class_targets, predictions, average='binary', zero_division=0)
            class_f1_score.append(f1)
        
        class_f1_scores.append(class_f1_score)
    
    return class_f1_scores




def main(opt):
    config_data = load_config(os.path.join(opt.model_directory, "config.json"))
    tokenizer = initialize_tokenizer(config_data.get("_name_or_path", None))
    id2_label = config_data.get("id2label", {})
    num_labels = len(id2_label)

    test_df, input_ids = prepare_test_data(opt.test_path, tokenizer)

    test_dataset = TestDataset(input_ids)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4, pin_memory=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(os.path.join(opt.model_directory), num_labels, device=device)
    targets = [TARGET_DICT[target] for target in test_df['category'].tolist()]

    if opt.uncertainty:
        all_probas, all_predictions, all_uncertainties = evaluate_model(model, test_loader, num_labels, device, True)
        all_uncertainties = np.array([uncertainty_item.cpu().item() for uncertainty_item in all_uncertainties])
        accuracy = sum([1 if all_predictions[i] == targets[i] else 0 for i in range(len(targets))]) / len(targets)

        uncertainty = np.mean(all_uncertainties)

        # Thresholds for accuracy/uncertainty graph
        thresholds = np.linspace(0, 1, 100)
        accuracies = [sum((all_uncertainties <= threshold) & (np.array(all_predictions) == np.array(targets))) / sum((all_uncertainties <= threshold)) for threshold in thresholds]
        logger.info("Uncertainty: %f", uncertainty)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, label='Accuracy')
        plt.axhline(y=accuracy, color='r', linestyle='--', label='Overall Accuracy')
        plt.xlabel('Uncertainty Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Uncertainty Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{opt.model_directory}/accuracy_uncertainty_threshold.png')
        plt.show()
    else:
        all_probas, all_predictions = evaluate_model(model, test_loader, num_labels, device)
        accuracy = sum([1 if all_predictions[i] == targets[i] else 0 for i in range(len(targets))]) / len(targets)


    precision = precision_score(targets, all_predictions, average='weighted')
    recall = recall_score(targets, all_predictions, average='weighted')
    f1 = f1_score(targets, all_predictions, average='weighted')

    logger.info("Accuracy: %f", accuracy)
    logger.info("Precision: %f", precision)
    logger.info("Recall: %f", recall)
    logger.info("F1 score: %f", f1)

    test_df['predicted'] = all_predictions
    test_df['probas'] = all_probas

    wrong_predictions = test_df[test_df['category'].map(TARGET_DICT) != test_df['predicted']]

    logger.info("Wrong Predictions:")
    logger.info(wrong_predictions)
    wrong_predictions.to_csv(f'{opt.model_directory}/wrong_predictions_test.csv', index=False)

    cm = confusion_matrix(targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(TARGET_INV_DICT.values()))
    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(ax=ax)
    plt.savefig(f'{opt.model_directory}/confusion_matrix_test.png')

    # Precision vs. Threshold graph
    all_thresholds = np.linspace(0, 1, 100)  # Define thresholds
    class_precisions = calculate_precision_at_thresholds(targets, all_probas, all_thresholds)

    # Calculate average precision across all classes
    avg_precision = np.mean(class_precisions, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot precision for each class
    for class_idx, class_precision in enumerate(class_precisions):
        class_name = id2_label[str(class_idx)]
        plt.plot(all_thresholds, class_precision, label=class_name)

    # Plot average precision curve
    plt.plot(all_thresholds, avg_precision, label="Average Precision", linestyle='--', color='black', linewidth=2)

    overall_precision = precision_score(targets, all_predictions, average='weighted')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs. Threshold for Each Class')
    plt.grid(True)
    plt.text(0.7, 0.1, f'Overall Precision: {overall_precision:.4f}', transform=plt.gca().transAxes)

    plt.savefig(f'{opt.model_directory}/precision_test.png')
    plt.show()

    # Recall vs. Threshold graph
    class_recalls = calculate_recall_at_thresholds(targets, all_probas, all_thresholds)

    avg_recalls = np.mean(class_recalls, axis=0)

    plt.figure(figsize=(10, 6))
    for class_idx, class_recall in enumerate(class_recalls):
        class_name = id2_label[str(class_idx)]
        plt.plot(all_thresholds, class_recall, label=class_name)

    plt.plot(all_thresholds, avg_recalls, label="Average Recall", linestyle='--', color='black', linewidth=2)

    overall_recall = recall_score(targets, all_predictions, average='weighted')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs. Threshold for Each Class')
    plt.grid(True)
    plt.text(0.7, 0.1, f'Overall Recall: {overall_recall:.4f}', transform=plt.gca().transAxes)

    plt.savefig(f'{opt.model_directory}/recall_test.png')
    plt.show()

    # Plot F1 score vs. threshold graph
    class_f1_scores = calculate_f1_at_thresholds(targets, all_probas, all_thresholds)
    avg_f1_scores = np.mean(class_f1_scores, axis=0)

    plt.figure(figsize=(10, 6))
    for class_idx, class_f1_score in enumerate(class_f1_scores):
        class_name = id2_label[str(class_idx)]
        plt.plot(all_thresholds, class_f1_score, label=class_name)

    plt.plot(all_thresholds, avg_f1_scores, label="Average F1 Score", linestyle='--', color='black', linewidth=2)

    overall_f1 = f1_score(targets, all_predictions, average='weighted')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold for Each Class')
    plt.grid(True)
    plt.text(0.7, 0.1, f'Overall F1 Score: {overall_f1:.4f}', transform=plt.gca().transAxes)

    plt.savefig(f'{opt.model_directory}/f1_score_test.jpeg')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the text_classification model")
    parser.add_argument("--test_path", required=True, type=str, help="Path to to-be-tested dataset")
    parser.add_argument("--model_directory", required=True, type=str, help="Path to classification model")
    parser.add_argument("--uncertainty", action="store_true", help="uncertainty")
    opt = parser.parse_args()
    main(opt)
