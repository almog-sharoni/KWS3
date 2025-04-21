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

def run_ablation_test(ablation_config, num_epochs=100):
    """
    Run ablation tests with different configurations.
    Args:
        ablation_config: Dict of parameters to ablate
        num_epochs: Number of epochs for each test
    """
    # Load datasets
    train_ds, val_ds, test_ds, silence_ds, info = load_speech_commands_dataset(reduced=False)
    bg_noise_ds = load_bg_noise_dataset()
    
    # Store results
    results = []

    # Log ablation start
    log_to_file("\n\n===== ABLATION STUDY STARTED =====\n")
    log_to_file(f"Testing configurations: {ablation_config}")
    
    for config_name, config_value in ablation_config.items():
        for value in config_value:
            # Log current test
            log_to_file(f"\n--- Testing {config_name} = {value} ---\n")
            
            # Create a copy of the original configs
            current_model_config = copy.deepcopy(model_config)
            current_dataset_config = copy.deepcopy(dataset)
            
            # Modify the configuration
            if config_name in current_model_config:
                current_model_config[config_name] = value
                log_to_file(f"Modified model config: {config_name} = {value}")
            elif config_name in current_dataset_config:
                current_dataset_config[config_name] = value
                log_to_file(f"Modified dataset config: {config_name} = {value}")
            elif config_name == "MFCC_transform":
                # Special case for MFCC transform
                log_to_file(f"Setting MFCC_transform = {value}")
            elif config_name == "derivative":
                # Special case for derivatives in MFCCs
                log_to_file(f"Setting derivative = {value}")
            elif config_name == "use_cls_token":
                # Special case for CLS token usage
                current_model_config["use_cls_token"] = value
                log_to_file(f"Setting use_cls_token = {value}")
            
            # Initialize datasets with configurations
            pytorch_train_dataset = TFDatasetAdapter(
                train_ds, 
                bg_noise_ds, 
                **current_dataset_config, 
                augmentation=[lambda x: add_time_shift_and_align(x)],
                MFCC_transform=True if "MFCC_transform" not in ablation_config else value if config_name == "MFCC_transform" else ablation_config["MFCC_transform"][0],
                derivative=True if "derivative" not in ablation_config else value if config_name == "derivative" else ablation_config["derivative"][0]
            )
            
            pytorch_val_dataset = TFDatasetAdapter(
                val_ds, 
                None, 
                **current_dataset_config, 
                augmentation=None,
                MFCC_transform=True if "MFCC_transform" not in ablation_config else value if config_name == "MFCC_transform" else ablation_config["MFCC_transform"][0],
                derivative=True if "derivative" not in ablation_config else value if config_name == "derivative" else ablation_config["derivative"][0]
            )
            
            # Create DataLoaders
            train_loader = DataLoader(pytorch_train_dataset, **data_loader, shuffle=True)
            val_loader = DataLoader(pytorch_val_dataset, **data_loader, shuffle=False)
            
            # Initialize test data for final evaluation
            pytorch_test_dataset = TFDatasetAdapter(
                test_ds, 
                None, 
                **current_dataset_config,
                augmentation=None,
                MFCC_transform=True if "MFCC_transform" not in ablation_config else value if config_name == "MFCC_transform" else ablation_config["MFCC_transform"][0],
                derivative=True if "derivative" not in ablation_config else value if config_name == "derivative" else ablation_config["derivative"][0]
            )
            test_loader = DataLoader(pytorch_test_dataset, **data_loader, shuffle=False)
            
            # Initialize model
            model = KeywordSpottingModel_with_cls(**current_model_config).to("cuda")
            
            # Loss function
            criterion = nn.CrossEntropyLoss().to("cuda")
            
            # Optimizer
            base_optimizer = optim.Adam(
                model.parameters(), 
                lr=optimizer_config['lr'], 
                weight_decay=optimizer_config['weight_decay']
            )
            optimizer = Lookahead(base_optimizer, **optimizer_config['lookahead'])
            
            # Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                **scheduler_config['reduce_lr_on_plateau']
            )
            
            # Model size
            model_size = sum(p.numel() for p in model.parameters())
            log_to_file(f"Model size: {model_size}")
            
            # Training loop
            try:
                train_accuracies, val_accuracies, train_losses, val_losses = trainig_loop(
                    model, 
                    num_epochs, 
                    train_loader, 
                    val_loader, 
                    criterion, 
                    optimizer, 
                    scheduler
                )
                
                # Final evaluation on test set
                model.eval()
                test_accuracy = 0
                test_total = 0
                
                with torch.no_grad():
                    for audio, labels in test_loader:
                        audio, labels = audio.to("cuda"), labels.to("cuda")
                        outputs = model(audio)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_accuracy += (predicted == labels).sum().item()
                
                test_accuracy_percentage = 100 * test_accuracy / test_total
                
                # Log test accuracy
                log_to_file(f"Test accuracy: {test_accuracy_percentage:.2f}%")
                
                # Save results
                result = {
                    'parameter': config_name,
                    'value': value,
                    'model_size': model_size,
                    'train_accuracy': train_accuracies[-1],
                    'validation_accuracy': val_accuracies[-1],
                    'test_accuracy': test_accuracy_percentage,
                    'train_loss': train_losses[-1],
                    'validation_loss': val_losses[-1]
                }
                
                results.append(result)
                
            except Exception as err:
                log_to_file(f"Error during training: {str(err)}")
                
    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv('ablation_results.csv', index=False)
    log_to_file("\n===== ABLATION STUDY COMPLETED =====\n")
    log_to_file(f"Results saved to ablation_results.csv")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation tests for Keyword Spotting model")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for each test")
    parser.add_argument("--test-mamba-layers", action="store_true",
                        help="Test different number of mamba layers")
    parser.add_argument("--test-dropout", action="store_true",
                        help="Test different dropout rates")
    parser.add_argument("--test-noise", action="store_true",
                        help="Test different noise levels")
    parser.add_argument("--test-mfcc", action="store_true",
                        help="Test with and without MFCC transform")
    parser.add_argument("--test-derivative", action="store_true",
                        help="Test with and without derivatives in MFCCs")
    parser.add_argument("--test-cls-token", action="store_true",
                        help="Test with and without CLS token")
    
    args = parser.parse_args()
    
    # Define ablation configurations
    ablation_config = {}
    
    if args.test_mamba_layers:
        ablation_config['num_mamba_layers'] = [1, 2, 3]
        
    if args.test_dropout:
        ablation_config['dropout_rate'] = [0.1, 0.2, 0.3, 0.5]
        
    if args.test_noise:
        ablation_config['noise_level'] = [0.0, 0.1, 0.2, 0.3]
        
    if args.test_mfcc:
        ablation_config['MFCC_transform'] = [True, False]
        
    if args.test_derivative:
        ablation_config['derivative'] = [True, False]
        
    if args.test_cls_token:
        ablation_config['use_cls_token'] = [True, False]
    
    # If no specific tests are selected, run a default set
    if len(ablation_config) == 0:
        ablation_config = {
            'num_mamba_layers': [1, 2, 4, 8, 16],
            'dropout_rate': [0.0, 0.5, 0.6],
            'noise_level': [0.0, 0.1, 0.2],
            'MFCC_transform': [True], 
            'derivative': [True, False],
            'use_cls_token': [True, False]
        }
    
    # Run the ablation tests
    run_ablation_test(ablation_config, num_epochs=args.epochs)