import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from collections import deque

class IncrementalLearner:
    def __init__(self, model: nn.Module, memory_size: int = 1000):
        self.model = model
        self.memory_size = memory_size
        self.memory_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.ewc_lambda = 0.4
        self.fisher_dict = {}
        self.optimal_params = {}

    def add_to_memory(self, features: np.ndarray, labels: np.ndarray):
        for f, l in zip(features, labels):
            self.memory_buffer.append((f, l))

    def sample_from_memory(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.memory_buffer) == 0:
            return None, None

        sample_size = min(batch_size, len(self.memory_buffer))
        indices = np.random.choice(len(self.memory_buffer), sample_size, replace=False)

        samples = [self.memory_buffer[i] for i in indices]
        features = torch.FloatTensor([s[0] for s in samples])
        labels = torch.FloatTensor([s[1] for s in samples])

        return features, labels

    def compute_fisher_information(self, dataloader):
        self.model.eval()
        fisher_dict = {}

        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param)

        for features, labels in dataloader:
            self.model.zero_grad()
            risk, _ = self.model(features)
            loss = nn.BCELoss()(risk.squeeze(), labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2

        for name in fisher_dict:
            fisher_dict[name] /= len(dataloader)

        self.fisher_dict = fisher_dict

        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def ewc_loss(self) -> torch.Tensor:
        if not self.fisher_dict:
            return torch.tensor(0.0)

        ewc_loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def incremental_update(
        self,
        new_features: np.ndarray,
        new_labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, float]:
        self.add_to_memory(new_features, new_labels)

        self.model.train()
        losses = []

        new_features_tensor = torch.FloatTensor(new_features)
        new_labels_tensor = torch.FloatTensor(new_labels)

        for epoch in range(epochs):
            epoch_losses = []

            for i in range(0, len(new_features_tensor), batch_size):
                batch_features = new_features_tensor[i:i+batch_size]
                batch_labels = new_labels_tensor[i:i+batch_size]

                memory_features, memory_labels = self.sample_from_memory(batch_size // 2)

                if memory_features is not None:
                    batch_features = torch.cat([batch_features, memory_features])
                    batch_labels = torch.cat([batch_labels, memory_labels])

                self.optimizer.zero_grad()

                risk, uncertainty = self.model(batch_features)

                task_loss = nn.BCELoss()(risk.squeeze(), batch_labels)

                regularization_loss = self.ewc_loss()

                total_loss = task_loss + regularization_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(total_loss.item())

            losses.append(np.mean(epoch_losses))

        return {
            'final_loss': losses[-1],
            'avg_loss': np.mean(losses),
            'epochs_trained': epochs,
            'samples_added': len(new_features)
        }

    def evaluate(self, test_features: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            features_tensor = torch.FloatTensor(test_features)
            predictions = self.model.predict_with_uncertainty(features_tensor, n_samples=30)

            risk_scores = predictions['risk_score']
            predicted_labels = (risk_scores > 0.5).astype(int)

            accuracy = (predicted_labels == test_labels).mean()

            tp = ((predicted_labels == 1) & (test_labels == 1)).sum()
            fp = ((predicted_labels == 1) & (test_labels == 0)).sum()
            fn = ((predicted_labels == 0) & (test_labels == 1)).sum()
            tn = ((predicted_labels == 0) & (test_labels == 0)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }

class OnlineMetaLearner:
    def __init__(self, model: nn.Module, meta_lr: float = 0.001, inner_lr: float = 0.01):
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr

    def adapt(self, support_features: torch.Tensor, support_labels: torch.Tensor, steps: int = 5):
        adapted_params = {}
        for name, param in self.model.named_parameters():
            adapted_params[name] = param.clone()

        for _ in range(steps):
            risk, _ = self.model(support_features)
            loss = nn.BCELoss()(risk.squeeze(), support_labels)

            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            for (name, param), grad in zip(self.model.named_parameters(), grads):
                adapted_params[name] = param - self.inner_lr * grad

        return adapted_params

    def meta_update(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        meta_loss = 0

        for support_x, support_y, query_x, query_y in task_batch:
            self.adapt(support_x, support_y)

            risk, _ = self.model(query_x)
            task_loss = nn.BCELoss()(risk.squeeze(), query_y)
            meta_loss += task_loss

        meta_loss /= len(task_batch)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
