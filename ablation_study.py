import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch_optimizer import Lookahead
from src.data.config import dataset, data_loader, model as model_config, optimizer as optimizer_config, scheduler as scheduler_config, training
import pandas as pd
import os
from src.models.model import KeywordSpottingModel_with_cls
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter, load_bg_noise_dataset
from src.utils.utils import print_model_size, log_to_file, plot_learning_curves
from src.utils.augmentations import add_time_shift_and_align, add_silence
from src.utils.train_utils import trainig_loop

def evaluate_model(model, test_loader, device):
    model.eval()
    accuracy = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in test_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    return 100 * accuracy / total

def load_existing_results(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def save_results(df, csv_path):
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

def run_ablation_study():
    # Load datasets
    train_ds, val_ds, test_ds, silence_ds, info = load_speech_commands_dataset(reduced=True)
    bg_noise_ds = load_bg_noise_dataset()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load existing results
    csv_path = 'ablation_study_results.csv'
    existing_results = load_existing_results(csv_path)
    print(f"Loaded {len(existing_results)} existing results")
    
    # Define models to study
    models_to_study = [
        {
            'name': 'RSM_Norm_0-1-2_order_cls_bgnoise (Baseline)',
            'path': 'notebooks/optuna/best_models/best_model_trial_66.pt',
            'config': {
                'd_state': 42,
                'd_conv': 33,
                'expand': 31,
                'num_mamba_layers': 7,
                'dropout_rate': 0.0318373730665146,
                'input_dim': model_config['input_dim'],
                'd_model': model_config['d_model'],
                'label_names': model_config['label_names']
            }
        },
        {
            'name': 'RSM_Norm_0-1-2_order_cls_bgnoise (No Noise)',
            'path': 'notebooks/optuna/best_models/best_model_trial_63.pt',
            'config': {
                'd_state': 42,
                'd_conv': 33,
                'expand': 31,
                'num_mamba_layers': 7,
                'dropout_rate': 0.0318373730665146,
                'input_dim': model_config['input_dim'],
                'd_model': model_config['d_model'],
                'label_names': model_config['label_names']
            }
        },
        {
            'name': 'RSM_Norm_0-1-2_order_cls_bgnoise (Reduced Layers)',
            'path': 'notebooks/optuna/best_models/best_model_trial_67.pt',
            'config': {
                'd_state': 42,
                'd_conv': 33,
                'expand': 31,
                'num_mamba_layers': 4,  # Reduced from 7
                'dropout_rate': 0.0318373730665146,
                'input_dim': model_config['input_dim'],
                'd_model': model_config['d_model'],
                'label_names': model_config['label_names']
            }
        },
        {
            'name': 'RSM_Norm_0-1-2_order_cls_bgnoise (Reduced State)',
            'path': 'notebooks/optuna/best_model_rsm_norm_reduced_state.pt',
            'config': {
                'd_state': 21,  # Reduced from 42
                'd_conv': 33,
                'expand': 31,
                'num_mamba_layers': 7,
                'dropout_rate': 0.0318373730665146,
                'n_mfcc': 46,
                'hop_length': 39,
                'input_dim': 46 * 3,
                'd_model': (16000 // 39) + 1,
                'noise_level': 0.0023867824085522
            }
        },
        {
            'name': 'RSM_Norm_0-1-2_order_cls_bgnoise (Reduced Conv)',
            'path': 'notebooks/optuna/best_model_rsm_norm_reduced_conv.pt',
            'config': {
                'd_state': 42,
                'd_conv': 16,  # Reduced from 33
                'expand': 31,
                'num_mamba_layers': 7,
                'dropout_rate': 0.0318373730665146,
                'n_mfcc': 46,
                'hop_length': 39,
                'input_dim': 46 * 3,
                'd_model': (16000 // 39) + 1,
                'noise_level': 0.0023867824085522
            }
        }
    ]
    
    results = []
    
    for model_info in models_to_study:
        model_name = model_info['name']
        
        # Check if results already exist for this model
        if not existing_results.empty and model_name in existing_results['Model'].values:
            print(f"\nSkipping {model_name} - results already exist")
            continue
            
        print(f"\nStudying {model_name}")
        print(f"Model dimensions: input_dim={model_info['config']['input_dim']}, d_model={model_info['config']['d_model']}")
        
        # Initialize datasets with configurations
        if 'No Noise' in model_name:
            pytorch_train_dataset = TFDatasetAdapter(train_ds, None, **dataset, augmentation=[lambda x: add_time_shift_and_align(x)])
        else:
            pytorch_train_dataset = TFDatasetAdapter(train_ds, bg_noise_ds, **dataset, augmentation=[lambda x: add_time_shift_and_align(x)])
        
        pytorch_val_dataset = TFDatasetAdapter(val_ds, None, **dataset, augmentation=None)
        pytorch_test_dataset = TFDatasetAdapter(test_ds, None, **dataset, augmentation=None)
        
        # Create DataLoaders
        train_loader = DataLoader(pytorch_train_dataset, **data_loader, shuffle=True)
        val_loader = DataLoader(pytorch_val_dataset, **data_loader, shuffle=False)
        test_loader = DataLoader(pytorch_test_dataset, **data_loader, shuffle=False)
        
        # Initialize model with specific config
        model = KeywordSpottingModel_with_cls(**model_info['config']).to(device)
        
        try:
            # Load pretrained weights
            model.load_state_dict(torch.load(model_info['path'], map_location=device))
            print(f"Successfully loaded weights from {model_info['path']}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_info['path']}: {str(e)}")
            print("Continuing with initialized weights...")
        
        # Calculate model size
        macs, params = print_model_size(model, input_size=torch.randn(1, model_info['config']['input_dim'], model_info['config']['d_model']-1).to(device))
        
        # Setup training components
        criterion = nn.CrossEntropyLoss().to(device)
        base_optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
        optimizer = Lookahead(base_optimizer, **optimizer_config['lookahead'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config['reduce_lr_on_plateau'])
        
        # Train with early stopping
        print("Starting training with early stopping...")
        train_accuracies, val_accuracies, train_losses, val_losses = trainig_loop(
            model=model,
            num_epochs=100,  # Set to 100 epochs
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_best_model=True,
            device=device
        )
        
        # Get final metrics
        final_train_accuracy = train_accuracies[-1]
        final_val_accuracy = val_accuracies[-1]
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        
        # Store results
        result = {
            'Model': model_name,
            'Train Loss': final_train_loss,
            'Train Accuracy (%)': final_train_accuracy,
            'Val Loss': final_val_loss,
            'Val Accuracy (%)': final_val_accuracy,
            'Test Accuracy (%)': test_accuracy,
            'Model Size (Params)': params,
            'MACs': macs,
            'd_state': model_info['config']['d_state'],
            'd_conv': model_info['config']['d_conv'],
            'expand': model_info['config']['expand'],
            'num_mamba_layers': model_info['config']['num_mamba_layers'],
            'dropout_rate': model_info['config']['dropout_rate'],
            'input_dim': model_info['config']['input_dim'],
            'd_model': model_info['config']['d_model'],
            'Best Val Accuracy (%)': max(val_accuracies),
            'Best Val Loss': min(val_losses)
        }
        
        results.append(result)
        
        # Save results after each model
        if not existing_results.empty:
            # Append new results to existing ones
            df = pd.concat([existing_results, pd.DataFrame([result])], ignore_index=True)
        else:
            # Create new DataFrame with results
            df = pd.DataFrame(results)
        
        save_results(df, csv_path)
        
        print(f"Training completed for {model_name}")
        print(f"Final Train Loss: {final_train_loss:.4f}")
        print(f"Final Train Accuracy: {final_train_accuracy:.2f}%")
        print(f"Final Val Loss: {final_val_loss:.4f}")
        print(f"Final Val Accuracy: {final_val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Model Size: {params:,} parameters")
        print(f"MACs: {macs/1e9:.2f} GMACs")

if __name__ == "__main__":
    run_ablation_study() 