import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
import argparse
import pandas as pd
from torch_optimizer import Lookahead
from src.data.config import dataset, data_loader, model as model_config, optimizer as optimizer_config, scheduler as scheduler_config, training

# Import custom modules
from src.models.model import KeywordSpottingModel_with_cls
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter, load_bg_noise_dataset
from src.utils.utils import set_memory_GB, print_model_size, log_to_file, plot_learning_curves
from src.utils.augmentations import add_time_shift_and_align, add_silence
from src.utils.train_utils import trainig_loop

# Add multiprocessing imports
import torch.multiprocessing as mp
import os
from functools import partial
import gc
import pickle
import glob
import time

# Define the top 3 models based on test accuracy and model size
TOP_MODELS = [
    {
        'trial_number': 43,
        'test_accuracy': 95.12,
        'model_size': 495274,
        'config': {
            'd_state': 9,
            'd_conv': 8,
            'expand': 2,
            'batch_size': 32,
            'dropout_rate': 0.35229388038128423,
            'num_mamba_layers': 3,
            'n_mfcc': 27,
            'n_fft': 323,
            'hop_length': 105,
            'n_mels': 35,
            'noise_level': 0.1404915536918465,
            'shift': 93,
            'lr': 0.0010935382053406887,
            'weight_decay': 9.957457594884688e-06
        }
    },
    {
        'trial_number': 67,
        'test_accuracy': 93.42,
        'model_size': 390236,
        'config': {
            'd_state': 8,
            'd_conv': 8,
            'expand': 2,
            'batch_size': 31,
            'dropout_rate': 0.38721803786212405,
            'num_mamba_layers': 4,
            'n_mfcc': 17,
            'n_fft': 484,
            'hop_length': 137,
            'n_mels': 39,
            'noise_level': 0.14498303198994286,
            'shift': 99,
            'lr': 0.0008431260922891309,
            'weight_decay': 8.7402876592472e-06
        }
    },
    {
        'trial_number': 70,
        'test_accuracy': 94.21,
        'model_size': 234706,
        'config': {
            'd_state': 21,
            'd_conv': 6,
            'expand': 2,
            'batch_size': 30,
            'dropout_rate': 0.3509578127765005,
            'num_mamba_layers': 4,
            'n_mfcc': 30,
            'n_fft': 304,
            'hop_length': 195,
            'n_mels': 45,
            'noise_level': 0.14823800885822133,
            'shift': 81,
            'lr': 0.0009903750118319871,
            'weight_decay': 6.40315597660744e-05
        }
    }
]

# Define ablation configurations
ABLATION_CONFIG = {
    'num_mamba_layers': [1, 2, 4, 8, 16],
    'dropout_rate': [0.0, 0.25, 0.5, 0.75],
    'noise_level': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'MFCC_transform': [True, False],
    'derivative': [True, False],
    'use_cls_token': [True, False]
}

def run_single_ablation(model_config, param_changed, original_value, new_value, shared_data, num_epochs=50, process_id=None):
    """
    Run a single ablation test
    Args:
        model_config: Configuration for the model
        param_changed: Name of the parameter being ablated
        original_value: Original value of the parameter
        new_value: New value to test
        shared_data: Dict containing shared datasets and other info
        num_epochs: Number of epochs for training
        process_id: Process ID for logging
    """
    # Set device for this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    torch.cuda.empty_cache()
    
    # Extract shared data
    train_ds = shared_data['train_ds']
    val_ds = shared_data['val_ds'] 
    test_ds = shared_data['test_ds']
    bg_noise_ds = shared_data['bg_noise_ds']
    
    # Log current test
    log_message = f"\n--- Process {process_id}: Testing {param_changed} = {new_value} (original: {original_value}) on {device} ---\n"
    print(log_message)
    log_to_file(log_message)
    
    # Create a copy of the original configs
    current_model_config = copy.deepcopy(model_config)
    current_dataset_config = copy.deepcopy(dataset)
    
    # Update the parameter being tested
    if param_changed in current_model_config:
        current_model_config[param_changed] = new_value
    elif param_changed in current_dataset_config:
        current_dataset_config[param_changed] = new_value
    
    # Initialize datasets with current configuration
    pytorch_train_dataset = TFDatasetAdapter(
        train_ds, 
        bg_noise_ds, 
        **current_dataset_config,
        augmentation=[lambda x: add_time_shift_and_align(x)]
    )
    pytorch_val_dataset = TFDatasetAdapter(
        val_ds, 
        None, 
        **current_dataset_config,
        augmentation=None
    )
    
    # Initialize test data for final evaluation
    pytorch_test_dataset = TFDatasetAdapter(
        test_ds, 
        None, 
        **current_dataset_config,
        augmentation=None
    )
    
    # Create data loaders
    current_data_loader = copy.deepcopy(data_loader)
    current_data_loader['batch_size'] = model_config.get('batch_size', data_loader['batch_size'])
    
    train_loader = DataLoader(pytorch_train_dataset, **current_data_loader, shuffle=True)
    val_loader = DataLoader(pytorch_val_dataset, **current_data_loader, shuffle=False)
    test_loader = DataLoader(pytorch_test_dataset, **current_data_loader, shuffle=False)
    
    # Initialize model with specific configuration
    model = KeywordSpottingModel_with_cls(**current_model_config).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Optimizer - use model's learning rate if available
    lr = current_model_config.get('lr', optimizer_config['lr'])
    weight_decay = current_model_config.get('weight_decay', optimizer_config['weight_decay'])
    
    base_optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    optimizer = Lookahead(base_optimizer, **optimizer_config['lookahead'])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        **scheduler_config['reduce_lr_on_plateau']
    )
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters())
    log_to_file(f"Process {process_id}: Model size: {model_size}")
    
    result = {}
    
    # Training loop
    try:
        train_accuracies, val_accuracies, train_losses, val_losses = trainig_loop(
            model, 
            num_epochs, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler,
            device=device
        )
        
        # Final evaluation on test set
        model.eval()
        test_accuracy = 0
        test_total = 0
        
        with torch.no_grad():
            for audio, labels in test_loader:
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_accuracy += (predicted == labels).sum().item()
        
        test_accuracy_percentage = 100 * test_accuracy / test_total
        
        # Log test accuracy
        log_to_file(f"Process {process_id}: Test accuracy: {test_accuracy_percentage:.2f}%")
        
        # Save results
        result = {
            'model_trial': current_model_config['trial_number'],
            'parameter': param_changed,
            'original_value': original_value,
            'new_value': new_value,
            'model_size': model_size,
            'train_accuracy': train_accuracies[-1],
            'validation_accuracy': val_accuracies[-1],
            'test_accuracy': test_accuracy_percentage,
            'train_loss': train_losses[-1],
            'validation_loss': val_losses[-1]
        }
        
    except Exception as err:
        log_to_file(f"Process {process_id}: Error during training: {str(err)}")
        result = {
            'model_trial': current_model_config['trial_number'],
            'parameter': param_changed,
            'original_value': original_value,
            'new_value': new_value,
            'error': str(err)
        }
    
    # Clean up memory
    del model, optimizer, criterion, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def run_parallel_ablation_study(num_epochs=50, num_processes=4):
    """
    Run ablation study in parallel using multiple processes on a single GPU
    Args:
        num_epochs: Number of epochs for each test
        num_processes: Number of parallel processes to run
    """
    # Load datasets once to share across processes
    train_ds, val_ds, test_ds, silence_ds, info = load_speech_commands_dataset(reduced=False)
    bg_noise_ds = load_bg_noise_dataset()
    
    # Create shared data dictionary
    shared_data = {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'bg_noise_ds': bg_noise_ds,
        'ablation_config': ABLATION_CONFIG
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('ablation_results', exist_ok=True)
    
    # Initialize results list
    all_results = []
    
    # Create process pool
    pool = mp.Pool(processes=num_processes)
    
    # Generate all ablation tasks
    tasks = []
    for model in TOP_MODELS:
        for param_name, values in ABLATION_CONFIG.items():
            original_value = model['config'].get(param_name)
            if original_value is not None:
                for value in values:
                    if value != original_value:  # Skip if value is same as original
                        tasks.append({
                            'model_config': model['config'],
                            'param_changed': param_name,
                            'original_value': original_value,
                            'new_value': value
                        })
    
    # Run ablation tests in parallel
    results = []
    for i, task in enumerate(tasks):
        process_id = i % num_processes
        result = pool.apply_async(
            run_single_ablation,
            args=(
                task['model_config'],
                task['param_changed'],
                task['original_value'],
                task['new_value'],
                shared_data,
                num_epochs,
                process_id
            )
        )
        results.append(result)
    
    # Collect results
    for result in results:
        try:
            result_data = result.get()
            all_results.append(result_data)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv('ablation_results/ablation_results.csv', index=False)
            
        except Exception as e:
            print(f"Error in ablation test: {str(e)}")
    
    # Close pool
    pool.close()
    pool.join()
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv('ablation_results/final_ablation_results.csv', index=False)
    
    # Log completion
    log_to_file("\n===== ABLATION STUDY COMPLETED =====\n")
    log_to_file(f"Results saved to ablation_results/final_ablation_results.csv")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run parallel ablation study on single GPU')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for each test')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes to run')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method to spawn
    mp.set_start_method('spawn', force=True)
    
    # Run ablation study
    results = run_parallel_ablation_study(args.num_epochs, args.num_processes) 