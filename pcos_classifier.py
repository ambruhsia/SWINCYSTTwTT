import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import numpy as np
import random

class ImageFolderEX(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class SwinPCOSClassifier:
    def __init__(self, train_dir, test_dir, enhanced_dir, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.enhanced_dir = enhanced_dir
        
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-4,
            weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6
        )
        
    def _create_model(self):
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=2
        )
        model = model.to(self.device)
        return model
    
    def prepare_data(self):
        train_dataset = ImageFolderEX(self.train_dir, transform=self.transform)
        enhanced_dataset = ImageFolderEX(self.enhanced_dir, transform=self.transform)
        test_dataset = ImageFolderEX(self.test_dir, transform=self.transform)
        
        combined_dataset = ConcatDataset([train_dataset, enhanced_dataset])
        
        self.train_loader = DataLoader(
            combined_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Training samples: {len(combined_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes: {train_dataset.classes}")
        
    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    def train(self, epochs=10):
        self.model.train()
        history = {
            'train_loss': [], 
            'train_acc': [],
            'learning_rates': []
        }
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss/(batch_idx+1),
                    'acc': 100.*correct/total,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            self.scheduler.step()
            
            epoch_loss = running_loss/len(self.train_loader)
            epoch_acc = 100.*correct/total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.save_model('best_swin_pcos_model.pth')
            
            print(f'\nEpoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')
        
        return history
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100.*correct/total
        test_loss = test_loss/len(self.test_loader)
        
        print(f'\nTest Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=['Not Infected', 'Infected'],
            digits=4
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Not Infected', 'Infected'],
            yticklabels=['Not Infected', 'Infected']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        print("\nClassification Report:")
        print(report)
        
        return test_accuracy, test_loss
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    classifier = SwinPCOSClassifier(
        train_dir="/kaggle/input/hmmmmm/data/train",
        test_dir="/kaggle/input/hmmmmm/data/test",
        enhanced_dir="/kaggle/working/enhanced_data",
        batch_size=12
    )
    
    classifier.prepare_data()
    history = classifier.train(epochs=8)
    test_accuracy, test_loss = classifier.evaluate()
