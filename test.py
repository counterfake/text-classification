import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tqdm

# Load the saved model from .bin file
model_path = "/home/counterfake/Teknofest2023/checkpoint/text_classification-dbmdz-bert-base-turkish-128k-uncased/pytorch_model.bin"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_state_dict = torch.load(model_path, map_location=device)

# Initialize the model and tokenizer
num_labels = 3  # change this to match the number of classes in your dataset
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-128k-uncased', num_labels=num_labels, state_dict=model_state_dict)
model.to(device)
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')

# Read the test data from CSV file
test_df = pd.read_csv('/home/counterfake/Teknofest2023/data/raw/train_3class_balanced.csv')
label2int = {'not-risky': 0, 'risky': 1, 'second hand': 2}
test_df['target'] = test_df['target'].map(label2int)

# Prepare the input texts for classification
input_texts = [str(text) for text in test_df['text'].tolist()]
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

# Run the input texts through the model for inference
model.eval()
with torch.no_grad():
    test_loader_len = len(test_loader)
    for batch in tqdm.tqdm(test_loader, total=test_loader_len, desc="Inference Progress"): 
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        if not 'all_logits' in locals():
            all_logits = logits.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), dim=0)

# Compute the predicted classes and the loss metrics
predicted_classes = all_logits.argmax(dim=-1).tolist()
targets = test_df['target'].tolist()
accuracy = sum([1 if predicted_classes[i] == targets[i] else 0 for i in range(len(targets))]) / len(targets)
precision = sum([1 if predicted_classes[i] == 1 and targets[i] == 1 else 0 for i in range(len(targets))]) / sum(predicted_classes)
recall = sum([1 if predicted_classes[i] == 1 and targets[i] == 1 else 0 for i in range(len(targets))]) / sum(targets)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the loss metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)

# Create a dataframe for wrong predictions
result_df = pd.DataFrame({'id': test_df['id'], 'text': test_df['text'], 'target': targets, 'predicted': predicted_classes})
result_df = result_df[result_df['target'] != result_df['predicted']]

# Save the wrong predictions to result.csv
result_df.to_csv("result.csv", index=False)
