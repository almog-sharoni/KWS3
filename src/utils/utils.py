import torch
import numpy as np
import thop
# from mamba_ssm import Mamba
import sys

sys.path.append('mambaPy/mamba.py')
# from mamba_simple import MambaQuantized as Mamba
import mambapy
from mambapy import mamba as Mamba

from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
from datetime import datetime

# Function to set the memory fraction for the current process
def set_memory_GB(GB=1):
    # Get the total memory of the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Calculate the fraction that corresponds to 1GB
    fraction = GB * 1024**3 / total_memory

    # Set the memory fraction for the current process
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"Memory fraction set to {fraction}")
    # Print fraction in GB
    print(f"Memory fraction in GB: {fraction * total_memory / 1024**3}")


def calculate_MAMBA_flops(layer, x, y):
    B, D, L = x[0].shape  # B: Batch size, D: d_model, L: Sequence length
    N = layer.d_state  # d_state from the layer

    # Calculate SSM-specific FLOPs
    ssm_flops = 27 * N * D * B * L  # 27 * d_state * d_model * batch_size * sequence_length

    # Calculate Linear Layer FLOPs (6 * D^2 * tokens for forward + backward pass)
    linear_flops = 6 * D * D * L  # 6 * d_model^2 * sequence_length (tokens)

    # Associative Scan FLOPs (6 * sequence_length)
    associative_scan_flops = 6 * L

    # Total FLOPs for the SSM layer
    total_flops = ssm_flops + linear_flops + associative_scan_flops

    # Add to the model's total_ops
    layer.total_ops += torch.DoubleTensor([total_flops])

    # Calculate the number of parameters
    params = sum(p.numel() for p in layer.parameters())

    # Add to the model's total_params
    layer.total_params += torch.DoubleTensor([params])

def print_model_size(model, input_size, verbose=False):
    macs, params, ret_layer_info = thop.profile(model, inputs=(input_size,)
    ,custom_ops={},report_missing=True and verbose, ret_layer_info=True)
    print()
    print(f"MACs: {macs} Which are {macs/1e9} Giga-MACs, Params: {params}")
    print()
    if verbose:
        print("Layer-wise information:")
        for layer, info in ret_layer_info.items():
            print(f"Layer: {layer}")
            print(f"Total FLOPs: {info[0]}, Total Params: {info[1]}")
            print()
    return macs, params


# Custom logging function to write to a text file with a timestamp
def log_to_file(message, filename="training_log.txt"):
    with open(filename, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

# Eraly stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered")
            return True
        return False
    
# Function for plotting learning curves
def plot_learning_curves(train_accuracies, val_accuracies, train_losses, val_losses, save_to_file=False):
  epochs = range(1,len(train_accuracies)+1)
  yticks = np.arange(0, 101, 5)
  
  plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
  plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.yticks(yticks)
  plt.grid(True)
  plt.show()
  if save_to_file:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.savefig(f'plots/learning_curves_acc{timestamp}.png')
  plt.clf()

  plt.plot(epochs, train_losses, 'r', label='Training Loss')
  plt.plot(epochs, val_losses, 'b', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.grid(True)
  if save_to_file:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.savefig(f'plots/learning_curves_loss{timestamp}.png')
  plt.show()
  
# Function to plot the distribution of labels in a dataset  
def compute_label_distribution(dataset,title="label_distribution", plot=False):
    # Step 1: Extract labels from the dataset
    labels = [element[1].numpy() for element in dataset]

    # Step 2: Count the occurrences of each label
    label_counts = Counter(labels)

    # Get the labels and their counts
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    if plot:
        # Step 3: Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title, size=20)
        plt.show()

    # return sorted dictionary
    return dict(sorted(label_counts.items()))


def save_model_results():
    macs, params = print_model_size(model,input_size=inputs)
    accuracy = max(val_accuracies)
    data = {'Model': ['KeywordSpottingModel'], 'GMACs': [macs], 'Params': [params], 'Accuracy': [accuracy]}
    model_config = {'input_dim': input_dim, 'd_model': d_model, 'd_state': d_state, 'd_conv': d_conv, 'expand': expand}
    data.update(model_config)
    df = pd.DataFrame(data, index=[0])
    df.to_csv('results.csv', mode='a', header=True)

def compute_inference_GPU_mem(model, input):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.max_memory_allocated()
    model(input)
    m1 = torch.cuda.max_memory_allocated()
    # Compute total memory used
    total_memory = (m1 - m0) / 1e6  # Convert to MB
    
    return total_memory