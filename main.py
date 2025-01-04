import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from scripts.data_loader import load_data
from scripts.utils import clean_text
from scripts.model import BERTClassifier
from scripts.training import train, evaluate

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def plot_metrics(metrics, title, ylabel, xlabel="Epochs", legend_labels=None):
    plt.figure(figsize=(8, 6))
    for metric in metrics:
        plt.plot(metric)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legend_labels:
        plt.legend(legend_labels)
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    args = parser.parse_args()

    class_columns = ["Materials", "Construction", "Color", "Finishing", "Durability"]
    data = load_data(args.data_path, class_columns)
    data['Review'] = data['Review'].apply(clean_text)

    texts = data['Review'].values
    labels = data[class_columns].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=128)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier('bert-base-uncased', num_classes=len(class_columns)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=4 * len(train_dataloader))

    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []

    for epoch in range(4):
        print(f"Epoch {epoch + 1}/4")
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        val_loss, val_accuracy, val_f1, val_hamming, val_report = evaluate(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Validation Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

    # Plot training and validation loss
    plot_metrics(
        [train_losses, val_losses],
        title="Training and Validation Loss",
        ylabel="Loss",
        legend_labels=["Training Loss", "Validation Loss"]
    )

    # Plot validation F1 score
    plot_metrics(
        [val_f1_scores],
        title="Validation F1 Score",
        ylabel="F1 Score",
        legend_labels=["Validation F1 Score"]
    )

if __name__ == "__main__":
    main()
