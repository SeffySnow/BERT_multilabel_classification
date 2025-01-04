import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions, actual_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.append(preds.cpu())
            actual_labels.append(labels.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    actual_labels = torch.cat(actual_labels, dim=0).numpy()

    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average="macro")
    hamming = hamming_loss(actual_labels, predictions)
    report = classification_report(actual_labels, predictions, zero_division=0)

    return total_loss / len(data_loader), accuracy, f1, hamming, report
