import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
# Initialize the model and tokenizer
num_labels = 2  # change this to match the number of classes in your dataset
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')

# Read the test data from CSV file
df = pd.read_csv('/home/counterfake/Teknofest2023/moncler_test.csv')

predicted_df = df[df['text'].str.contains('is a post', na=False) | df['text'].isna()]
df.drop(predicted_df.index, inplace=True)

# Prepare the input texts for classification
input_texts = [str(text) for text in df['text'].tolist()]
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
model_path = "/home/counterfake/Teknofest2023/checkpoint/risky_two_class_2-dbmdz-bert-base-turkish-128k-uncased/pytorch_model.bin"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_state_dict = torch.load(model_path, map_location=device)
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-128k-uncased', num_labels=num_labels, state_dict=model_state_dict)
model.to(device)
model.eval()

all_logits = None
with torch.no_grad():
  for batch in test_loader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    if all_logits is None:
      all_logits = logits.detach().cpu()
    else:
      all_logits = torch.cat((all_logits, logits.detach().cpu()), dim=0)

# Compute the predicted classes and the loss metrics
predicted_classes = all_logits.argmax(dim=-1).tolist()
df['predicted'] = predicted_classes
df = df[['id', 'predicted']]
predicted_df = predicted_df[['id']].copy()
predicted_df['predicted'] = 0
df = pd.concat([df, predicted_df], axis=0)

df.to_csv("moncler_predictions.csv", index=False)