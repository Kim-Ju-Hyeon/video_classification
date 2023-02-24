import os
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim

from utils.train_helper import model_snapshot, load_model

from utils.score import get_score
import yaml
from utils.train_helper import edict2dict

from models.base_cnn import Base_CNN


class Runner(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.seed = config.seed
        self.device = config.device

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')
        self.metrics_file = os.path.join(config.exp_sub_dir, 'results.json')


        self.train_conf = config.train
        self.dataset_conf = config.dataset
        
        # Binary or Multi-Class Classification
        if self.train_conf.loss_type == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.train_conf.loss_type == 'multi_class':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported Loss")

        # Choose the model
        if self.config.model_name == 'base_cnn':
            self.model = Base_CNN(n_classes=13)
        else:
            raise ValueError("Non-supported Model")
        
        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.train_conf.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum)
        elif self.train_conf.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        self.model = self.model.to(device=self.device)
        self.criterion = self.criterion.to(device=self.device)
        
    def train(self, train_dataloader, val_dataloader):
        best_val_loss = float('inf')
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        for epoch in range(1, self.train_conf.epoch+1):
            # Train the model for one epoch
            train_loss, train_acc = self._train_epoch(train_dataloader)

            # Compute validation loss and accuracy
            val_loss, val_acc = self._evaluate(val_dataloader)

            # Log the results for this epoch
            self.logger.info(f'Epoch {epoch:03d}: train_loss = {train_loss:.4f} | train_acc = {train_acc:.4f} | '
                            f'val_loss = {val_loss:.4f} | val_acc = {val_acc:.4f}')
            
            self.logger.info(f'Best validation loss: {best_val_loss:.4f}')

            # Save the training and validation metrics for this epoch
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Save the weights of the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)
            # Save the training and validation metrics to a file
            metrics = {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f)
        
        
    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_samples += targets.size(0)

            # Log the loss for this batch
            total_loss += loss.item()

        # Compute average loss and accuracy for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc
    
    def _evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Compute accuracy
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)

                # Compute loss
                total_loss += loss.item()

        # Compute average loss and accuracy for the validation set
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc
    
    def test(self, dataloader):
        self.model.eval()

        # Load the weights of the best model
        self.model.load_state_dict(torch.load(self.best_model_dir))

        # Evaluate the model on the test set
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                predictions.append(preds.cpu().numpy())

        predictions = np.concatenate(predictions)
        return predictions