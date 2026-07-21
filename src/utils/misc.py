import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

## data augmentation of NIR data, add slope, offset, noise and shift
class data_augmentation:
    def __init__(self, slope = 0.1, offset = 0.1, noise = 0.1, shift = 0.1):
        self.slope = slope
        self.offset = offset
        self.noise = noise
        self.shift = shift

    def __call__(self, X):
        X_aug = np.zeros_like(X)
        X_aug = X * (1 + np.random.uniform(-self.slope, self.slope)) + np.random.uniform(-self.offset, self.offset) + np.random.normal(0, self.noise, len(X))
        return X_aug


class TrainerConfig:
    def __init__(self,model_name,  project_root=None):
        """
        Configuration class for training the model.

        :param project_root: Absolute path to the project's root directory. If None, it is dynamically determined.
        """
        if project_root is None:
            project_root = self.find_project_root(Path(__file__).resolve().parent)

        self.project_root = project_root
        self.save_path = self.project_root /"models" / model_name

        # Training hyperparameters (editable)
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.classification = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validation and saving options (editable)
        self.save_model = True
        self.max_loss_plot = 10

    def find_project_root(self, start_path, marker="src"):
        """
        Searches for the project root by traversing up the directory tree looking for a specific marker (e.g., 'src').

        :param start_path: The starting path for the search.
        :param marker: The specific marker to identify the root (default is 'src').
        :return: The path to the project root or None if not found.
        """
        current_path = Path(start_path).resolve()
        while current_path != current_path.root:
            if (current_path / marker).exists():
                return current_path
            current_path = current_path.parent
        return None

    def __repr__(self):
        """
        String representation of the configuration, mainly for debugging purposes.
        """
        return f"TrainerConfig(batch_size={self.batch_size}, learning_rate={self.learning_rate}, " \
               f"num_epochs={self.num_epochs}, device={self.device}, save_model={self.save_model}, "\
                f" classification={self.classification}, max_loss_plot={self.max_loss_plot},"\
                f"save_path={self.save_path})"

    def update_config(self, batch_size=None, learning_rate=None, num_epochs=None, save_model=None,
                       classification=None, max_loss_plot=None,save_path=None):
        """
        Method to update configuration parameters dynamically.

        :param batch_size: New batch size (optional).
        :param learning_rate: New learning rate (optional).
        :param num_epochs: New number of epochs (optional).
        :param save_model: Whether to save the model (optional).
        :param save_interval: Interval at which the model is saved (optional).
        :param early_stop: Whether early stopping is enabled (optional).
        :param save_best_only: Whether to save only the best model (optional).
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if save_model is not None:
            self.save_model = save_model
        if classification is not None:
            self.classification = classification
        if max_loss_plot is not None:
            self.max_loss_plot = max_loss_plot
        if save_path is not None:
            self.save_path = save_path


class Utils:

    @staticmethod
    def save_model(model, path, epoch, best_metric, current_metric ,classification, verbose=True):

        if best_metric == -np.inf:
            torch.save(model.state_dict(), path)
            print(f'Model saved at epoch {epoch} to {path}') if verbose else None

        current_metric = np.array(current_metric)
        if np.mean(current_metric) > 0:

            current_mean = np.mean(current_metric)
            best_mean = best_metric if isinstance(best_metric, float) else np.mean(best_metric)
            condition = current_mean > best_mean

            if condition:
                torch.save(model.state_dict(), path)
                print(f'Model saved at epoch {epoch} to {path}') if verbose else None
                best_metric = current_metric  # Update best_metric

        return best_metric

    @staticmethod
    def plot_losses(train_losses, val_losses, val_metrics, classification=False,maxplot_loss = 10):
        """
        Plot training and validation losses and metrics with improved formatting.
        """
        train_losses_np = [loss.numpy() for loss in train_losses]
        val_losses_np = [loss.numpy() for loss in val_losses]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Colors
        train_color = '#1f77b4'  # blue
        val_color = '#ff7f0e'  # orange
        metric_color = '#2ca02c'  # green

        # Plot losses
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=train_color)
        ax1.plot(train_losses_np, label='Training Loss', color=train_color, linewidth=2)
        ax1.plot(val_losses_np, label='Validation Loss', color=val_color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=train_color)
        ax1.set_ylim(0, min(maxplot_loss, max(max(train_losses_np), max(val_losses_np)) * 1.1))  # Cap loss axis to 10

        # Legends for loss
        ax1.legend(loc='upper left')

        # Metrics on second axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Metric', color=metric_color)
        ax2.tick_params(axis='y', labelcolor=metric_color)

        if not classification:
            for i in range(len(val_metrics[0])):
                metric_scores = [scores[i] for scores in val_metrics]
                ax2.plot(metric_scores, label=f'R² Score y{i}', linestyle='--', linewidth=2, color = metric_color)
                ax2.set_ylim(0, 1)  #
        else:
            ax2.plot(val_metrics, label='F1 Score', linestyle='--', color=metric_color, linewidth=2)

        ax2.legend(loc='upper right')

        plt.title('Training & Validation Loss and Metrics')
        fig.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show(block=False)
