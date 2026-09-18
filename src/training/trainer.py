import torch
import torcheval.metrics
from src.utils import Utils
import numpy as np
import os
import json
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, config, verbose=True,
                 early_stopping_patience=None, early_stopping_min_delta=0.0, trial=None):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            config: TrainerConfig object
            verbose: Print progress
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum change for early stopping
            trial: Optuna trial for pruning
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.best_val_metric = -np.inf
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.verbose = verbose
        self.best_epoch = None

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self._best_monitor_metric = -np.inf
        self._epochs_no_improve = 0

        # Optuna trial
        self.trial = trial

        # Checkpointing
        self.checkpoint_dir = None
        self.checkpoint_path = None
        self.best_checkpoint_path = None
        self.metadata_path = None
        
        # Setup checkpoint if save_path is provided
        if hasattr(config, 'save_path') and config.save_path is not None:
            self.checkpoint_dir = Path(config.save_path).parent
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
            self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
            self.metadata_path = self.checkpoint_dir / "metadata.json"
        
        # Cosine annealing LR scheduler
        self.scheduler = None
        if hasattr(config, 'use_cosine_lr') and config.use_cosine_lr:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )

    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_epoch': self.best_epoch,
        }
        
        # Save checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'best_val_metric': float(self.best_val_metric) if self.best_val_metric != -np.inf else None,
            'best_epoch': self.best_epoch,
            'checkpoint_path': str(self.checkpoint_path),
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if is_best and self.best_checkpoint_path:
            torch.save(self.model.state_dict(), self.best_checkpoint_path)
            
        if self.verbose:
            print(f'Checkpoint saved at epoch {epoch}')

    def load_checkpoint(self):
        """Load training checkpoint if exists."""
        if not self.checkpoint_dir or not self.checkpoint_path or not self.checkpoint_path.exists():
            return None
            
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_metric = checkpoint.get('best_val_metric', -np.inf)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_metrics = checkpoint.get('val_metrics', [])
            self.best_epoch = checkpoint.get('best_epoch', None)
            
            if self.verbose:
                print(f'Resumed from epoch {checkpoint["epoch"] + 1}')
            return checkpoint['epoch']
        except Exception as e:
            if self.verbose:
                print(f'Could not load checkpoint: {e}')
            return None

    def train_one_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = torch.zeros(1, device=self.device)
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True)
            if not self.config.classification:
                targets = targets.float()
            else:
                targets = targets.long()

            self.optimizer.zero_grad()
            outputs = self.model(inputs[:, None])
            loss = self.criterion(outputs, targets)
            loss.mean().backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            running_loss += torch.mean(loss) * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        val_loss = torch.zeros(1, device=self.device)
        out, tar = [], []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True).float()
                targets = targets.to(self.device, non_blocking=True)
                if not self.config.classification:
                    targets = targets.float()
                else:
                    targets = targets.long()
                outputs = self.model(inputs[:, None])

                loss = self.criterion(outputs, targets)
                val_loss += loss.mean() * inputs.size(0)

                out.append(outputs.detach().cpu())
                tar.append(targets.detach().cpu())

        val_loss = val_loss / len(self.val_loader.dataset)
        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)

        metrics = self.compute_metrics(all_outputs, all_targets)
        return val_loss, metrics

    def compute_metrics(self, outputs, targets):
        """Compute evaluation metrics."""
        metrics = []
        if not self.config.classification:
            if hasattr(self.model, 'out_dims'):
                n_outputs = self.model.out_dims
            else:
                n_outputs = outputs.shape[1]
                
            R2 = [torcheval.metrics.R2Score() for _ in range(n_outputs)]
            for i in range(n_outputs):
                R2[i].update(outputs[:, i], targets[:, i])
                metrics.append(R2[i].compute().item())
        else:
            F1 = torcheval.metrics.MulticlassF1Score()
            target_labels = (
                torch.argmax(targets, dim=1)
                if targets.ndim > 1
                else targets.long()
            )
            F1.update(target_labels, torch.argmax(outputs, dim=1))
            metrics = F1.compute()
        return metrics

    def train(self):
        """Main training loop with checkpointing and early stopping."""
        # Try to resume from checkpoint
        start_epoch = 0
        if self.checkpoint_path and self.checkpoint_path.exists():
            resumed_epoch = self.load_checkpoint()
            if resumed_epoch is not None:
                start_epoch = resumed_epoch + 1

        for epoch in range(start_epoch, self.config.num_epochs):
            # Train
            epoch_train_loss = self.train_one_epoch()
            self.train_losses.append(epoch_train_loss.detach().cpu())
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Evaluate
            val_loss, metrics = self.evaluate()
            self.val_losses.append(val_loss.detach().cpu())
            self.val_metrics.append(metrics)
            
            # Compute current metric
            current_monitor_metric = float(np.mean(np.atleast_1d(metrics)))
            
            # Check if best
            is_best = current_monitor_metric > self.best_val_metric + 1e-6
            if is_best:
                self.best_val_metric = current_monitor_metric
                self.best_epoch = epoch

            # Verbose output
            if self.verbose:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                      f"LR: {lr:.6f} | "
                      f"Train Loss: {epoch_train_loss[0].detach().cpu().numpy():.4f} | "
                      f"Val Loss: {val_loss[0].detach().cpu().numpy():.4f} | "
                      f"Val Metric: {current_monitor_metric:.4f}")

            # Save checkpoint (every epoch)
            if self.checkpoint_dir:
                self.save_checkpoint(epoch, is_best=is_best)

            # Optuna pruning
            if self.trial is not None:
                self.trial.report(current_monitor_metric, epoch)
                if self.trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

            # Early stopping
            if current_monitor_metric > self._best_monitor_metric + self.early_stopping_min_delta:
                self._best_monitor_metric = current_monitor_metric
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1

            if self.early_stopping_patience is not None and self._epochs_no_improve >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Plot losses
        if self.verbose:
            Utils.plot_losses(self.train_losses, self.val_losses, self.val_metrics, 
                            self.config.classification, self.config.max_loss_plot)

        return self.train_losses, self.val_losses, self.val_metrics