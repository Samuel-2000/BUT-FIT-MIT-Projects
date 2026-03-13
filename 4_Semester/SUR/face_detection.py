import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse

# Dataset pre trénovanie a validáciu
class PersonIdentificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: int(cls_name) - 1 for cls_name in self.classes}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file in os.listdir(cls_dir):
                if file.endswith('.png'):
                    self.image_files.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Dataset pre evaluáciu
class EvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        return image, filename


# CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=31):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Tréningová funkcia
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {100*correct/total:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'face_detection.pth')

    print(f"Best validation accuracy: {best_val_acc:.2f}%")


# Evaluačná funkcia
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


# Generovanie výsledkov pre test
def generate_eval_results(model, eval_loader, device, output_file):
    model.eval()
    with open(output_file, 'w') as f, torch.no_grad():
        for images, filenames in eval_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)
            predicted_classes = predicted_classes + 1  # späť na 1–31

            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)

            for i in range(len(filenames)):
                scores = log_probs[i]
                scores_str = " ".join([f"{score:.6f}" for score in scores])
                f.write(f"{filenames[i]} {predicted_classes[i].item()} {scores_str}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_dir', type=str, default='eval')
    parser.add_argument('--output_file', type=str, default='face_detection_predictions.txt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if args.train:
        print("Loading training and validation data...")
        train_dataset = PersonIdentificationDataset('train', transform)
        val_dataset = PersonIdentificationDataset('dev', transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = CNN(num_classes=31).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, val_loader, criterion, optimizer, device)

    if args.eval:
        print("Evaluating...")
        model = CNN(num_classes=31).to(device)
        model.load_state_dict(torch.load('face_detection.pth'))

        eval_dataset = EvalDataset(args.eval_dir, transform)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

        generate_eval_results(model, eval_loader, device, args.output_file)
        print(f"Results written to {args.output_file}")
