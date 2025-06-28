import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import torch.nn.utils.prune as prune
import torch.quantization
import onnxruntime

# --- 1. Setup ---
torch.backends.quantized.engine = 'fbgemm'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Preprocessing and Dataset ---
def preprocess_tweet(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

class TwitterDataset(Dataset):
    def __init__(self, texts, topic_ids, hate_labels, tokenizer, max_len):
        self.texts = texts
        self.topic_ids = topic_ids
        self.hate_labels = hate_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'topic_id': torch.tensor(self.topic_ids[item], dtype=torch.long),
            'hate_label': torch.tensor(self.hate_labels[item], dtype=torch.long)
        }

# --- 3. Data Loading ---
try:
    df = pd.read_csv('processed-data/processed_dataset.csv')
    print("Loaded existing processed_dataset.csv")
except FileNotFoundError:
    print("processed_dataset.csv not found. Generating dummy data.")
    data = {
        'Text': [
            "This is a normal tweet about sports.", "I hate that group of people, they are terrible!",
            "Women should stay in the kitchen #gender", "Vote for our candidate, best for the country #politics",
            "All people of color are lazy #race", "Religion is the opiate of the masses. #religion",
            "This is a great day!", "Some very offensive comments here! lol",
            "Racist remarks are not tolerated.", "Boys will be boys. #gender",
            "Political debate getting heated.", "The match was fantastic!",
            "I despise their beliefs. #religion", "Another day, another sexist remark. #gender",
            "Equality for all races!", "Voting is important.", "Winning team deserved it!",
            "This is just normal talk. 😇", "Some slang: lmao rofl",
            "Check out this cool link: https://example.com/xyz",
        ],
        'TopicID': [4, 2, 1, 3, 2, 0, 4, 1, 2, 1, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0],
        'HateLabel': [0, 2, 2, 0, 2, 1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)

df['Processed_Text'] = df['Text'].apply(preprocess_tweet)

X_train, X_test, y_train, y_test, topic_train, topic_test = train_test_split(
    df['Processed_Text'], df['HateLabel'], df['TopicID'],
    test_size=0.2, random_state=42, stratify=df['HateLabel']
)

# --- 4. Model and Dataloader Setup ---
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3

tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME, num_labels=3
)
model.to(device)

def create_data_loader(texts, topics, labels, tokenizer, max_len, batch_size, shuffle=True):
    ds = TwitterDataset(
        texts=texts.to_list(),
        topic_ids=topics.to_list(),
        hate_labels=labels.to_list(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

train_data_loader = create_data_loader(X_train.reset_index(drop=True), topic_train.reset_index(drop=True), y_train.reset_index(drop=True), tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(X_test.reset_index(drop=True), topic_test.reset_index(drop=True), y_test.reset_index(drop=True), tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

# --- 5. Quantization-Aware Training (QAT) Setup ---
model.train()
torch.quantization.prepare_qat(model, inplace=True)
print("Model prepared for Quantization-Aware Training (QAT).")
optimizer = AdamW(model.parameters(), lr=2e-5)

# --- 6. Training and Evaluation Loops ---
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["hate_label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model = model.eval()
    all_preds, all_labels, all_topic_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            labels, topic_ids = batch["hate_label"].to(device), batch["topic_id"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_topic_ids.extend(topic_ids.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_topic_ids)

# --- 7. Main Training Execution ---
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_loss = train_epoch(model, train_data_loader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")

# --- 8. Final Evaluation and Fairness Metrics ---
print("\n--- Final Evaluation and Fairness Metrics ---")
y_pred, y_true, topic_ids = eval_model(model, test_data_loader, device)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Offensive', 'Hate'], zero_division=0))

eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'topic_id': topic_ids})
eval_df['y_true_binary'] = eval_df['y_true'].apply(lambda x: 0 if x == 0 else 1)
eval_df['y_pred_binary'] = eval_df['y_pred'].apply(lambda x: 0 if x == 0 else 1)
eval_df['is_gender_topic'] = (eval_df['topic_id'] == 1).astype(int)
eval_df['is_race_topic'] = (eval_df['topic_id'] == 2).astype(int)

protected_attributes = ['is_gender_topic', 'is_race_topic']
for attr in protected_attributes:
    print(f"\n--- Fairness Metrics for Protected Attribute: '{attr}' ---")
    aif_df = eval_df.copy()
    unprivileged_groups = [{attr: 1}]
    privileged_groups = [{attr: 0}]
    
    if not (1 in aif_df[attr].unique() and 0 in aif_df[attr].unique()):
        print(f"Skipping fairness metrics for '{attr}': The test set does not contain both privileged and unprivileged groups.")
        continue
    try:
        # Favorable label is 0 ('Normal'), unfavorable is 1 ('Offensive' or 'Hate').
        dataset_true = BinaryLabelDataset(
            df=aif_df,
            label_names=['y_true_binary'],
            protected_attribute_names=[attr],
            favorable_label=0,
            unfavorable_label=1
        )
        dataset_pred = dataset_true.copy()
        dataset_pred.labels = aif_df['y_pred_binary'].values.reshape(-1, 1)
        metric = ClassificationMetric(
            dataset_true,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        spd = metric.statistical_parity_difference()
        print(f"Statistical Parity Difference: {spd:.4f}")
        di = metric.disparate_impact()
        print(f"Disparate Impact: {di:.4f}")
        aod = metric.average_odds_difference()
        print(f"Average Odds Difference: {aod:.4f}")
    except Exception as e:
        print(f"Could not calculate fairness metrics for '{attr}' due to an error: {e}")

# --- 9. Post-Training Optimization (Pruning & Quantization) ---
print("\n--- Applying Pruning & Finalizing Quantization ---")
model.to('cpu')
model.eval()

print("Applying Pruning...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.random_unstructured(module, name="weight", amount=0.5)
        prune.remove(module, 'weight')

print("Converting to fully quantized model...")
model_quantized = torch.quantization.convert(model, inplace=False)

# --- 10. Export to ONNX ---
class ModelForOnnx(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask).logits

onnx_model_path = "hate_speech_model_quantized_pruned.onnx"
model_for_export = ModelForOnnx(model_quantized)
dummy_input = (
    torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN), dtype=torch.long),
    torch.ones((1, MAX_LEN), dtype=torch.long)
)
dynamic_axes = {'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

print("\n--- Exporting model to ONNX ---")
torch.onnx.export(
    model_for_export, dummy_input, onnx_model_path,
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'], output_names=['output'],
    dynamic_axes=dynamic_axes
)
print(f"Model exported to {onnx_model_path}")

# --- 11. Verify ONNX Model and Run Prediction ---
def predict_onnx(text, tokenizer, ort_session, max_len=128):
    text = preprocess_tweet(text)
    encoding = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=max_len, return_token_type_ids=False,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    ort_inputs = {
        ort_session.get_inputs()[0].name: encoding['input_ids'].cpu().numpy(),
        ort_session.get_inputs()[1].name: encoding['attention_mask'].cpu().numpy()
    }
    logits = ort_session.run(None, ort_inputs)[0]
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    prediction = np.argmax(probabilities)
    label_map = {0: 'Normal', 1: 'Offensive', 2: 'Hate'}
    return label_map[prediction], probabilities

try:
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    print("\nONNX model loaded successfully and verified.")
    test_tweet = "Women are weak and should not be leaders."
    prediction, probs = predict_onnx(test_tweet, tokenizer, ort_session)
    print(f"\nTest Tweet: '{test_tweet}'")
    print(f"Prediction: {prediction}, Probabilities: {probs}")
except Exception as e:
    print(f"\nError during ONNX verification or prediction: {e}")

print("\n--- Script Finished ---")