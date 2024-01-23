import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
from utils.constants import TARGET_INV_DICT
import json
import os

parser = argparse.ArgumentParser(description="Test the text_classification model")
parser.add_argument("--test_path", required=True, type=str, help="Path to to-be-tested dataset")
parser.add_argument("--model_directory", required=True, type=str, help="Path to classifcation model")

opt = parser.parse_args()

# Get Config
config_path =  os.path.join(opt.model_directory, "config.json")
with open(config_path, 'r') as json_file:
    config_data = json.load(json_file)

# Extract configs
tokenizer_name = config_data.get("_name_or_path", None)
num_of_classes = len(config_data.get("id2label", {}))

# Initiliaze Tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# Read the test data from CSV file
test_df = pd.read_csv(opt.test_path)

# Prepare the input texts for classification
input_texts = [str(text) for text in test_df['description_text'].tolist()]
input_ids = tokenizer.batch_encode_plus(input_texts, add_special_tokens=True, truncation=True, padding='max_length', return_tensors='pt')

# Create a dataset and data loader for the test set
class TestDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids['input_ids']
        self.attention_mask = input_ids['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

test_dataset = TestDataset(input_ids)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4, pin_memory=True)

# Load the saved models and evaluate on the test set
model_path = os.path.join(opt.model_directory, "pytorch_model.bin")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_state_dict = torch.load(model_path, map_location=device)
model = BertForSequenceClassification.from_pretrained(opt.tokenizer, num_labels=num_labels, state_dict=model_state_dict)
model.to(device)
model.eval()

all_logits = None
all_probas = None
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # 'logits' attribute instead of 'outputs[0]'
        probas = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
        if all_logits is None:
            all_logits = logits.detach().cpu()
            all_probas = probas.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), dim=0)
            all_probas = torch.cat((all_probas, probas.detach().cpu()), dim=0)


# Compute the predicted classes and the loss metrics
predicted_classes = all_logits.argmax(dim=-1).tolist()
targets = test_df['category'].tolist()

# Map predicted classes to target values
predicted_classes = [TARGET_INV_DICT[p] for p in predicted_classes]

accuracy = sum([1 if predicted_classes[i] == targets[i] else 0 for i in range(len(targets))]) / len(targets)
precision = precision_score(targets, predicted_classes, average='weighted')  # Specify 'average' parameter
recall = recall_score(targets, predicted_classes, average='weighted')  # Specify 'average' parameter
f1 = f1_score(targets, predicted_classes, average='weighted')  # Specify 'average' parameter

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


test_df['predicted'] = predicted_classes
probas_list = all_probas.tolist()
predicted_probs = [sorted(zip(range(num_labels), probas), key=lambda x: x[1], reverse=True) for probas in probas_list]
top_predictions = [(TARGET_INV_DICT[pred], proba) for pred, proba in predicted_probs[0]]
test_df['probas'] = predicted_probs

wrong_predictions = test_df[test_df['category'] != test_df['predicted']]

# Add probabilities to the DataFrame

print("Wrong Predictions:")
print(wrong_predictions)
wrong_predictions.to_csv('wrong_predictions_test.csv', index=False)
# Compute the confusion matrix and display
cm = confusion_matrix(targets, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(TARGET_INV_DICT.values()))
# Get the figure and axis
fig, ax = plt.subplots(figsize=(20, 20))

# Plot the confusion matrix on the specified axis
disp.plot(ax=ax)
plt.savefig('confusion_matrix_test.png')
