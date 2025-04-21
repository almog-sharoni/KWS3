#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import sys
sys.path.append("../../")

from torch_optimizer import Lookahead


from src.models.model import KeywordSpottingModel_with_cls
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter, load_bg_noise_dataset
from src.utils.utils import set_memory_GB, print_model_size, log_to_file, plot_learning_curves, EarlyStopping
from src.utils.augmentations import add_time_shift_and_align, add_silence
from src.utils.train_utils import trainig_loop




# In[2]:


torch.cuda.is_available()


# In[ ]:


train_ds, val_ds, test_ds, silence_ds , info = load_speech_commands_dataset(reduced=True)


# In[4]:


bg_noise_ds = load_bg_noise_dataset()


# In[5]:


print(train_ds)


# In[6]:


# maintain seed for repructablity
np.seed = 42
# tf.random.set_seed(42)
torch.manual_seed(0)


# In[7]:


label_names = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
print(label_names)


# In[8]:


# augmentations = [
#     lambda x: add_time_shift_and_align(x),
# ]


# In[9]:


# # Convert the TFDS dataset to a PyTorch Dataset
# fixed_length = 16000
# n_mfcc = 13
# n_fft = 400
# hop_length = 160
# n_mels = 40
# pytorch_train_dataset = TFDatasetAdapter(train_ds, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentations)
# pytorch_val_dataset = TFDatasetAdapter(val_ds, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentations=None)


# In[10]:


# # Create a DataLoader to feed the data into the model
# batch_size = 32
# train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2)
# val_loader = DataLoader(pytorch_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,prefetch_factor=2)


# In[11]:


# for audio, label in train_loader:
#     print(audio.shape, label.shape)
#     break


# In[12]:


# # Varify Tensor's shape
# # Example audio sample
# from librosa.feature import mfcc
# audio = np.random.randn(16000).astype(np.float32)  # Simulate 1-second audio at 16kHz

# # Compute MFCC features
# mfcc_features = mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40, fmin=0, fmax=8000)

# print(f'MFCC shape: {mfcc_features.shape}')  # Expected: (13, num_frames)


# In[ ]:





# # Training loop

# # With L2 regulariztion AND Droput layer

# In[13]:


import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from math import ceil
import gc  # For garbage collection
import numpy as np  # Ensure numpy is imported
import csv  # For CSV file writing
import os   # For checking if CSV file exists

# Import your custom modules and classes here
# from your_module import TFDatasetAdapter, EarlyStopping, Lookahead, KeywordSpottingModel_with_cls, add_time_shift_and_align, label_names

def log_to_file(message, filename):
    with open(filename, 'a') as f:
        f.write(message + "\n")

def objective(trial):
    # Log trial start and current GPU memory usage
    log_to_file(f"--- Starting trial {trial.number} ---", "optuna.log")
    log_to_file(f"Trial params: {trial.params}", "optuna.log")
    try:
        log_to_file(torch.cuda.memory_summary(device='cuda'), "optuna.log")
    except Exception as mem_e:
        log_to_file(f"Memory summary error: {mem_e}", "optuna.log")

    model = None
    optimizer = None
    criterion = None
    train_loader = None
    val_loader = None
    test_loader = None
    test_accuracy = 0  # Default test_accuracy for later CSV output

    # Initialization block
    try:
        np.random.seed(42)
        torch.manual_seed(0)

        # Suggest hyperparameters
        d_state = trial.suggest_int('d_state', 8, 64)
        d_conv = trial.suggest_int('d_conv', 2, 16)
        expand = trial.suggest_int('expand', 2, 4)
        batch_size = trial.suggest_int('batch_size', 8, 64)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.7)
        num_mamba_layers = trial.suggest_int('num_mamba_layers', 1, 4)
        n_mfcc = trial.suggest_int('n_mfcc', 6, 26)
        n_fft = trial.suggest_int('n_fft', 200, 800)
        hop_length = trial.suggest_int('hop_length', 20, 640)
        n_mels = trial.suggest_int('n_mels', 20, 100)
        noise_level = trial.suggest_float('noise_level', 0.0, 0.3)
        shift = trial.suggest_int('shift', 1, 150)

        # Define augmentations
        augmentations = [
            lambda x: add_time_shift_and_align(x, shift),
        ]

        # Convert the TFDS dataset to a PyTorch Dataset.
        # Note: Ensure train_ds, val_ds, test_ds, and bg_noise_ds are defined or loaded appropriately.
        fixed_length = 16000
        pytorch_train_dataset = TFDatasetAdapter(
            train_ds, bg_noise_ds, fixed_length, n_mfcc, n_fft, hop_length, n_mels,
            augmentation=augmentations, noise_level=noise_level
        )
        pytorch_val_dataset = TFDatasetAdapter(
            val_ds, None, fixed_length, n_mfcc, n_fft, hop_length, n_mels,
            augmentation=None
        )

        # Define DataLoader for training and validation
        train_loader = DataLoader(
            pytorch_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            pytorch_val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2
        )

        # Init early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        # Initialize the model
        model = KeywordSpottingModel_with_cls(
            input_dim=n_mfcc * 3,
            d_model=(16000 // hop_length) + 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout_rate=dropout_rate,
            label_names=label_names,
            num_mamba_layers=num_mamba_layers
        ).to("cuda")

        # Define optimizer and learning rate scheduler
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
        base_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # Define loss function
        criterion = nn.CrossEntropyLoss().to("cuda")

        log_to_file(f"Starting training with hyperparameters: {trial.params}", "optuna.log")

    except Exception as init_e:
        log_to_file(f"Initialization error: {init_e}", "optuna.log")
        return 0.0, 0

    num_epochs = 100
    for epoch in range(num_epochs):
        try:
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            batch_count = 0

            for audio, labels in train_loader:
                try:
                    # Move data to CUDA with CUDA OOM handling
                    try:
                        audio, labels = audio.to("cuda"), labels.to("cuda")
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            log_to_file("CUDA OOM during training data transfer", "optuna.log")
                            torch.cuda.empty_cache()
                            gc.collect()
                            return 0.0, 0
                        else:
                            log_to_file(f"RuntimeError: {e}", "optuna.log")
                            raise

                    optimizer.zero_grad()
                    outputs = model(audio)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    batch_count += 1

                except Exception as batch_e:
                    log_to_file(f"Epoch {epoch} training batch error: {batch_e}", "optuna.log")
                    continue

            if batch_count > 0:
                train_loss = running_loss / batch_count
                train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
            else:
                train_loss = 0
                train_accuracy = 0

            log_to_file(f"Epoch {epoch} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%", "optuna.log")

            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            val_batch_count = 0

            with torch.no_grad():
                for audio, labels in val_loader:
                    try:
                        try:
                            audio, labels = audio.to("cuda"), labels.to("cuda")
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                log_to_file("CUDA OOM during validation data transfer", "optuna.log")
                                torch.cuda.empty_cache()
                                gc.collect()
                                return 0.0, 0
                            else:
                                log_to_file(f"RuntimeError: {e}", "optuna.log")
                                raise

                        outputs = model(audio)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                        val_batch_count += 1

                    except Exception as val_batch_e:
                        log_to_file(f"Epoch {epoch} validation batch error: {val_batch_e}", "optuna.log")
                        continue

            if val_batch_count > 0:
                val_loss_avg = val_loss / val_batch_count
                val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0
            else:
                val_loss_avg = 0
                val_accuracy = 0

            log_to_file(f"Epoch {epoch} - Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%", "optuna.log")

            # # --- Pruning ---
            # trial.report(val_accuracy, step=epoch)
            # if trial.should_prdune():
            #     log_to_file(f"Trial {trial.number} pruned at epoch {epoch}", "optuna.log")
            #     raise optuna.TrialPruned()

            # Step the scheduler based on validation loss
            try:
                scheduler.step(val_loss_avg)
            except Exception as sched_e:
                log_to_file(f"Epoch {epoch} scheduler error: {sched_e}", "optuna.log")

            # Early stopping check
            if early_stopping.step(val_loss_avg):
                log_to_file(f"Stopping training at epoch {epoch+1} due to early stopping", "optuna.log")
                break

            # --- Test Phase ---
            try:
                pytorch_test_dataset = TFDatasetAdapter(
                    test_ds, None, fixed_length, n_mfcc, n_fft, hop_length, n_mels,
                    augmentation=None
                )
                test_loader = DataLoader(
                    pytorch_test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=2
                )
            except Exception as test_loader_e:
                log_to_file(f"Epoch {epoch} test loader creation error: {test_loader_e}", "optuna.log")
                continue

            correct_test = 0
            total_test = 0
            test_batch_count = 0
            model.eval()
            with torch.no_grad():
                for audio, labels in test_loader:
                    try:
                        try:
                            audio, labels = audio.to("cuda"), labels.to("cuda")
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                log_to_file("CUDA OOM during test data transfer", "optuna.log")
                                torch.cuda.empty_cache()
                                gc.collect()
                                return 0.0, 0
                            else:
                                log_to_file(f"RuntimeError: {e}", "optuna.log")
                                raise

                        outputs = model(audio)
                        _, predicted = torch.max(outputs, 1)
                        total_test += labels.size(0)
                        correct_test += (predicted == labels).sum().item()
                        test_batch_count += 1

                    except Exception as test_batch_e:
                        log_to_file(f"Epoch {epoch} test batch error: {test_batch_e}", "optuna.log")
                        continue

            test_accuracy = 100 * correct_test / total_test if total_test > 0 else 0
            log_to_file(f"Epoch {epoch} - Test Accuracy: {test_accuracy:.2f}%", "optuna.log")

        except Exception as epoch_e:
            log_to_file(f"Error in epoch {epoch}: {epoch_e}", "optuna.log")
            continue

    # --- Final Evaluation ---
    try:
        model_size = sum(p.numel() for p in model.parameters())
        log_to_file(f"Final test accuracy: {test_accuracy:.2f}%", "optuna.log")
        log_to_file(f"Final validation accuracy: {val_accuracy:.2f}%, Model size: {model_size}", "optuna.log")
    except Exception as final_e:
        log_to_file(f"Final evaluation error: {final_e}", "optuna.log")
        model_size = 0

    # Save high accuracy results to CSV if applicable
    if test_accuracy > 88:
        try:
            file_name = "high_accuracy_results.csv"
            result_row = {
                "trial": trial.number,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "val_accuracy": val_accuracy,
                "model_size": model_size,
                "model_path": f"model_{trial.number}.pth"
            }
            result_row.update(trial.params)
            file_exists = os.path.isfile(file_name)
            with open(file_name, "a", newline="") as csvfile:
                fieldnames = list(result_row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result_row)
        except Exception as csv_ex:
            log_to_file(f"Error writing to CSV: {csv_ex}", "optuna.log")

    try:
        return val_accuracy, model_size
    except Exception as ret_e:
        log_to_file(f"Error returning metrics: {ret_e}", "optuna.log")
        return 0.0, 0

    finally:
        # Cleanup Section: Free CUDA memory and other resources
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if criterion is not None:
            del criterion
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        if test_loader is not None:
            del test_loader

        torch.cuda.empty_cache()
        gc.collect()
        try:
            log_to_file(torch.cuda.memory_summary(device="cuda"), "optuna.log")
        except Exception as mem_e:
            log_to_file(f"Memory summary error during cleanup: {mem_e}", "optuna.log")
        log_to_file(f"--- Ending trial {trial.number} ---", "optuna.log")




# In[ ]:


# from optuna import samplers.NSGAIISampler

sampler = optuna.samplers.NSGAIISampler()

# Define the storage location (a SQLite file)
storage_name = "sqlite:///optuna_study.db"
study_name = "KWS_study_new"

# Create a multi-objective study
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=sampler,  # Optional: Choose a sampler suitable for multi-objective
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True
)

# Optimize the study with catch for Exception and TrialPruned
i = 0
while i < 1000:
    try:
        study.optimize(objective, n_trials=1000, show_progress_bar=False, n_jobs=8,
                       catch=(Exception, optuna.TrialPruned))
        i += 1
    except Exception as e:
        log_to_file(f"Error: {e}", "optuna.log")
        continue


# In[21]:


from optuna.visualization import plot_optimization_history

fig = plot_optimization_history(study)
fig.show()


# In[22]:


from optuna.visualization import plot_param_importances

fig = plot_param_importances(study)
fig.show()


# In[23]:


from optuna.visualization import plot_parallel_coordinate

fig = plot_parallel_coordinate(study)
fig.show()


# In[24]:


from optuna.visualization import plot_slice

fig = plot_slice(study)
fig.show()


# In[ ]:


# print study name
print(study.study_name)


# In[ ]:


# save the study
import joblib
joblib.dump(study, 'study.pkl')


# In[ ]:


study.best_params


# In[4]:


torch.cuda.empty_cache()


# In[6]:


import pandas as pd
import torch
from utils import compute_inference_GPU_mem, print_model_size
from src.models.model import KeywordSpottingModel_with_cls  # Import your model class

# Load the CSV file
df_filtered = pd.read_csv('filtered_model_training_data.csv')

# Iterate through the rows in the dataframe and process each configuration
results = []
for index, row in df_filtered.iterrows():  # Use .iterrows() to correctly iterate over rows
    # Clear cuda
    torch.cuda.empty_cache()
    input_dim =  int(row['n_mfcc']) * 3  # Assuming n_mfcc as input dimension
    d_model = int(16000 / row['hop_length'] + 1 + 1)  # Correctly access d_model from the row
    d_state =  int(row['d_state'])
    d_conv =  int(row['d_conv'])
    expand =  int(row['expand'])
    num_mamba_layers =  int(row['num_mamba_layers'])
    dropout_rate = int(row['dropout_rate'])
    label_names = ['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10', 'label11', 'label12']  # Adjust based on your labels
    
    # Create the model
    model = KeywordSpottingModel_with_cls(
        input_dim=input_dim,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        label_names=label_names,
        num_mamba_layers=num_mamba_layers,
        dropout_rate=dropout_rate
    ).to("cuda")
    
    # Get batch size from the config
    batch_size = int(row['batch_size'])
    
    # Calculate model size (MACs, params) and accuracy
    macs, params = print_model_size(model, input_size=torch.randn(batch_size, input_dim, d_model-1).to("cuda"))
    # other metrics
    train_accuracy = row['train_accuracy']
    validation_accuracy = row['validation_accuracy']
    test_accuracy = row['test_accuracy']
    training_epochs = row['epochs']
    batch_size = row['batch_size']
    lr = row['lr']
    weight_decay = row['weight_decay']
    noise_level = row['noise_level']

    
    # Compute inference GPU memory usage
    inf_GPU_mem = compute_inference_GPU_mem(model, input=torch.randn(1, input_dim, d_model-1).to("cuda"))
    
    # Calculate inference MACs and params
    inf_macs, inf_params = print_model_size(model, input_size=torch.randn(1, input_dim, d_model-1).to("cuda"))
    
    # Store the results for this model configuration
    result = {
        'Model': 'KeywordSpottingModel_RSM_Norm_0-1-2_order_cls_bgnoise',
        'Training MMACs': macs / 1e6,
        'KParams': params / 1e3,
        'Train Accuracy': train_accuracy,
        'Validation Accuracy': validation_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Epochs': training_epochs,
        'Batch Size': batch_size,
        'Learning Rate': lr,
        'Weight Decay': weight_decay,
        'Noise Level': noise_level,
        'Inference CUDA Mem in MB': inf_GPU_mem,
        'Inference MMACs': inf_macs / 1e6,
        'Inference KParams': inf_params / 1e3,
        'input_dim': input_dim,
        'd_model': d_model,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand
    }
    
    results.append(result)

# Convert results to a DataFrame and save them
df_results = pd.DataFrame(results)
df_results.to_csv('results2.csv', mode='a', header=True, index=False)

print("Results have been saved to results2.csv")


# In[10]:


import pandas as pd
import torch
from src.utils.utils import compute_inference_GPU_mem, print_model_size
from src.models.model import KeywordSpottingModel_with_cls  # Import your model class
import torch
import gc
import pandas as pd
import torch
import gc
import pandas as pd

# Initialize the results list
results = []

# Iterate over each row in the DataFrame
for index, row in df_filtered.iterrows():
    # Extract parameters from the current row
    input_dim = int(row['n_mfcc']) * 3
    d_model = int(16000 / row['hop_length'] + 2)  # Simplified
    d_state = int(row['d_state'])
    d_conv = int(row['d_conv'])
    expand = int(row['expand'])
    num_mamba_layers = int(row['num_mamba_layers'])
    dropout_rate = float(row['dropout_rate'])  # Changed to float
    batch_size = int(row['batch_size'])
    train_accuracy = row['train_accuracy']
    validation_accuracy = row['validation_accuracy']
    test_accuracy = row['test_accuracy']
    training_epochs = row['epochs']
    lr = row['lr']
    weight_decay = row['weight_decay']
    noise_level = row['noise_level']
    
    label_names = [
        'label1', 'label2', 'label3', 'label4', 'label5', 'label6',
        'label7', 'label8', 'label9', 'label10', 'label11', 'label12'
    ]
    
    # Initialize variables to None
    model = None
    input_tensor = None
    inf_input = None

    try:
        with torch.no_grad():
            # Create and move the model to GPU
            model = KeywordSpottingModel_with_cls(
                input_dim=input_dim,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                label_names=label_names,
                num_mamba_layers=num_mamba_layers,
                dropout_rate=dropout_rate
            ).to("cuda")
            
            # Generate input tensor for model size computation
            input_tensor = torch.randn(batch_size, input_dim, d_model - 1, device="cuda")
            macs, params = print_model_size(model, input_size=input_tensor, verbose=True)
            
            # Compute inference GPU memory usage
            inf_input = torch.randn(1, input_dim, d_model - 1, device="cuda")
            inf_GPU_mem = compute_inference_GPU_mem(model, input=inf_input)
            
            # Calculate inference MACs and params
            inf_macs, inf_params = print_model_size(model, input_size=inf_input)
            
            # Store the results for this model configuration
            result = {
                'Model': 'KWS3',
                'Training MMACs': macs / 1e6,
                'KParams': params / 1e3,
                'Train Accuracy': train_accuracy,
                'Validation Accuracy': validation_accuracy,
                'Test Accuracy': test_accuracy,
                'Training Epochs': training_epochs,
                'Batch Size': batch_size,
                'Learning Rate': lr,
                'Weight Decay': weight_decay,
                'Noise Level': noise_level,
                'Inference CUDA Mem in MB': inf_GPU_mem,
                'Inference MMACs': inf_macs / 1e6,
                'Inference KParams': inf_params / 1e3,
                'input_dim': input_dim,
                'd_model': d_model,
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand
            }
            
            results.append(result)
            
    except Exception as e:
        print(f"Error processing row {index}: {e}")
    
    finally:
        # Cleanup Section
        if model is not None:
            del model
        if input_tensor is not None:
            del input_tensor
        if inf_input is not None:
            del inf_input
        torch.cuda.empty_cache()
        gc.collect()

# Convert results to a DataFrame and save them
df_results = pd.DataFrame(results)
df_results.to_csv('results2.csv', mode='a', header=True, index=False)

print("Results have been saved to results2.csv")


# In[ ]:




