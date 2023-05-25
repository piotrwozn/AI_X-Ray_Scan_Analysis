import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class XrayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.labels_frame = df['Finding Labels'].str.get_dummies('|')
        self.image_frame = df['Image Index']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx])
        image = Image.open(img_name)
        labels = self.labels_frame.iloc[idx].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


with open('train_val_list.txt', 'r') as file:
    train_val_list = [line.strip() for line in file]

with open('test_list.txt', 'r') as file:
    test_list = [line.strip() for line in file]

df_test = pd.read_csv('Data_Entry_2017.csv')
df_test = df_test[df_test['Image Index'].isin(test_list)]

df = pd.read_csv('Data_Entry_2017.csv')
df = df[df['Image Index'].isin(train_val_list)]

train_df, val_df = train_test_split(df, test_size=0.2)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = XrayDataset(df=train_df, root_dir='images/', transform=train_transform)
val_dataset = XrayDataset(df=val_df, root_dir='images/', transform=val_transform)
test_dataset = XrayDataset(df=df_test, root_dir='images/', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_name('efficientnet-b6')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(train_dataset[0][1]))

model = model.to(device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

patience = 5
best_val_loss = float('inf')
early_stop_counter = 0


def calculate_accuracy(real, predict):
    predict = predict > 0.5
    real = real.to(torch.uint8)
    corrects = (predict == real).sum().item()
    total = real.numel()
    return corrects / total


def calculate_recall(real, predict):
    predict = predict > 0.5
    real = real.to(torch.uint8)
    true_positive = (predict & real).sum().item()
    false_negative = ((~predict) & real).sum().item()

    if true_positive + false_negative > 0:
        return true_positive / (true_positive + false_negative)
    else:
        return 1


def calculate_f1(precision, recall):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def calculate_precision(real, predict):
    predict = predict > 0.5
    real = real.to(torch.uint8)
    true_positive = (predict & real).sum().item()
    false_positive = (predict & (~real)).sum().item()

    if true_positive + false_positive > 0:
        return true_positive / (true_positive + false_positive)
    else:
        return 1


if os.path.exists('model_1.pth'):
    model.load_state_dict(torch.load('model_1.pth'))
    print('Model loaded from model_1.pth')

for epoch in range(50):
    print(f'Epoch {epoch + 1}/{50}')
    print('-' * 10)

    # Trenowanie modelu
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0

    pbar = tqdm(train_loader, total=len(train_loader), desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(labels, torch.sigmoid(outputs))
        precision = calculate_precision(labels, torch.sigmoid(outputs))
        recall = calculate_recall(labels, torch.sigmoid(outputs))
        f1 = calculate_f1(precision, recall)

        running_acc += acc
        running_loss += loss.item() * inputs.size(0)
        running_precision += precision
        running_recall += recall
        running_f1 += f1

        pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=100 * running_acc / (pbar.n + 1),
                         f1=running_f1 / (pbar.n + 1), recall=running_recall / (pbar.n + 1),
                         precision=running_precision / (pbar.n + 1))

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_acc / len(train_dataset)
    epoch_f1 = running_f1 / len(train_loader)
    print(
        f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {100 * epoch_acc:.2f}%, Training F1: {100 * epoch_f1:.2f}%')

    # Walidacja
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    pbar = tqdm(val_loader, total=len(val_loader), desc="Validation")
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            acc = calculate_accuracy(labels, torch.sigmoid(outputs))
            precision = calculate_precision(labels, torch.sigmoid(outputs))
            recall = calculate_recall(labels, torch.sigmoid(outputs))
            f1 = calculate_f1(precision, recall)

            val_acc += acc
            val_loss += loss.item() * inputs.size(0)
            val_precision += precision
            val_recall += recall
            val_f1 += f1

            pbar.set_postfix(loss=val_loss / (pbar.n + 1), acc=100 * val_acc / (pbar.n + 1),
                             prec=val_precision / (pbar.n + 1), f1=val_f1 / (pbar.n + 1))

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_acc / len(val_dataset)
    epoch_val_prec = val_precision / len(val_loader)
    epoch_val_f1 = val_f1 / len(val_loader)
    print(
        f'Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {100 * epoch_val_acc:.2f}%, Validation Precision: {100 * epoch_val_prec:.2f}%, Validation F1: {100 * epoch_val_f1:.2f}%')

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early Stopping: No improvement in validation loss. Training stopped.")
        break

    torch.save(model.state_dict(), f'model_{epoch + 1}.pth')

class_names = test_dataset.labels_frame.columns.tolist()

# Testowanie modelu
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

        predicted_classes = [class_name for class_name, pred in zip(class_names, predictions) if pred]
        true_classes = [class_name for class_name, true_label in zip(class_names, labels.squeeze().numpy()) if
                        true_label]

        filename = test_dataset.image_frame.iloc[i]

        print(f'File: {filename}, Predicted classes: {predicted_classes}, True classes: {true_classes}')
