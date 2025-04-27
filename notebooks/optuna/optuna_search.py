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
# from src.utils.augmentations import add_time_shift_and_align, add_silence
import src.utils.augmentations as augmentations
from src.utils.train_utils import trainig_loop



torch.cuda.is_available()
# ─── load raw TF-DS splits ────────────────────────────────────────────────────
train_tf, val_tf, test_tf, _, info = load_speech_commands_dataset(reduced=False)


# ─── load background noise dataset ─────────────────────────────────────────────
bg_noise_tf = load_bg_noise_dataset()          # ← original TF/NumPy
bg_noise_ds = [
    torch.from_numpy(n.numpy()).float()        # ⚠ adjust if already np.ndarray
    if hasattr(n, "numpy") else torch.from_numpy(n).float()
    for n in bg_noise_tf
]
# build default aug objects ONE time; reuse inside every trial
wave_aug, spec_aug = augmentations.build_default_augs(bg_noise_ds)  # returns two callables

# maintain seed for repructablity
np.seed = 42
# tf.random.set_seed(42)
torch.manual_seed(0)
label_names = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'silence', 'unknown']
print(label_names)
# augmentations = [
#     lambda x: add_time_shift_and_align(x),
# ]
# # Convert the TFDS dataset to a PyTorch Dataset
# fixed_length = 16000
# n_mfcc = 13
# n_fft = 400
# hop_length = 160
# n_mels = 40
# pytorch_train_dataset = TFDatasetAdapter(train_ds, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentations)
# pytorch_val_dataset = TFDatasetAdapter(val_ds, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentations=None)
# # Create a DataLoader to feed the data into the model
# batch_size = 32
# train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2)
# val_loader = DataLoader(pytorch_val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,prefetch_factor=2)
# for audio, label in train_loader:
#     print(audio.shape, label.shape)
#     break

# # Varify Tensor's shape
# # Example audio sample
# from librosa.feature import mfcc
# audio = np.random.randn(16000).astype(np.float32)  # Simulate 1-second audio at 16kHz

# # Compute MFCC features
# mfcc_features = mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40, fmin=0, fmax=8000)

# print(f'MFCC shape: {mfcc_features.shape}')  # Expected: (13, num_frames)


# Training loop
# With L2 regulariztion AND Droput layer
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
        d_state = trial.suggest_int('d_state', 16, 128)
        d_conv = trial.suggest_int('d_conv', 2, 64)
        expand = trial.suggest_int('expand', 2, 4)
        batch_size = trial.suggest_int('batch_size', 8, 64)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.5)
        num_mamba_layers = trial.suggest_int('num_mamba_layers', 1, 8)
        n_mfcc = trial.suggest_int('n_mfcc', 6, 40)
        n_fft = trial.suggest_int('n_fft', 50, 800)
        hop_length = trial.suggest_int('hop_length', 20, 640)
        n_mels = trial.suggest_int('n_mels', 20, 100)
        # noise_level = trial.suggest_float('noise_level', 0.0, 0.8)
        # shift = trial.suggest_int('shift', 1, 150)



        # Convert the TFDS dataset to a PyTorch Dataset.
        # Note: Ensure train_ds, val_ds, test_ds, and bg_noise_ds are defined or loaded appropriately.
        fixed_len = 16000
    # ── build PyTorch datasets (adapter will do padding + aug) ─────────────
        train_ds = TFDatasetAdapter(
            train_tf,
            fixed_length=fixed_len,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            waveform_augs=[wave_aug],        # <- list!
            spec_augs=[spec_aug],
            derivative=True,                 # keep Δ and Δ²
            compute_mfcc=True,
        )
        val_ds = TFDatasetAdapter(
            val_tf,
            fixed_length=fixed_len,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            waveform_augs=[],                # no aug at val / test
            spec_augs=[],
            derivative=True,
            compute_mfcc=True,
        )

        # ── build inverse-frequency weights for a WeightedRandomSampler ──────────
        label_hist = np.bincount([int(l.numpy()) for _, l in train_tf], minlength=12)
        inv_freq   = 1.0 / (label_hist + 1e-9)
        sample_wts = [inv_freq[int(l.numpy())] for _, l in train_tf]

        sampler = torch.utils.data.WeightedRandomSampler(
            sample_wts, num_samples=len(train_tf), replacement=True
        )
        
        # ── data loaders ───────────────────────────────────────────────────────
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,
            num_workers=4, prefetch_factor=2, pin_memory=True,
            persistent_workers=True 
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, prefetch_factor=2
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
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        base_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # Define loss function
        criterion = nn.CrossEntropyLoss().to("cuda")

        log_to_file(f"Starting training with hyperparameters: {trial.params}", "optuna.log")

    except Exception as init_e:
        log_to_file(f"Initialization error: {init_e}", "optuna.log")
        return 0.0

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
                            return 0.0
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
                                return 0.0
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

            # --- Pruning ---
            trial.report(val_accuracy, step=epoch)
            if trial.should_prune():
                log_to_file(f"Trial {trial.number} pruned at epoch {epoch}", "optuna.log")
                raise optuna.TrialPruned()

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
                    test_tf,
                    fixed_length=fixed_len,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    waveform_augs=[],
                    spec_augs=[],
                    derivative=True,
                    compute_mfcc=True,
                )
                test_loader = DataLoader(
                    pytorch_test_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=0
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
                                return 0.0
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

        except optuna.TrialPruned:
            log_to_file(f"Trial {trial.number} pruned at epoch {epoch}", "optuna.log")
            # return here so Optuna marks the trial as finished
            return val_accuracy

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
        return val_accuracy
    except Exception as ret_e:
        log_to_file(f"Error returning metrics: {ret_e}", "optuna.log")
        return 0.0

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




sampler = optuna.samplers.TPESampler(
    multivariate=True,         # Better modeling of interactions between hyperparameters
    group=True,                # Groups categorical variables (e.g., optimizers)
    n_startup_trials=100        # You can increase if you have larger search space
)

# Define the storage location (a SQLite file)
storage_name = "sqlite:///optuna_study.db"
study_name = "KWS_study_new"

# Create a multi-objective study
study = optuna.create_study(
    directions=["maximize"],
    sampler=sampler,  # Optional: Choose a sampler suitable for multi-objective
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True
)

# Optimize the study with catch for Exception and TrialPruned
i = 0
while i < 1000:
    try:
        study.optimize(objective, n_trials=1000, show_progress_bar=False, n_jobs=1,
                       catch=(Exception, optuna.TrialPruned))
        i += 1
    except Exception as e:
        log_to_file(f"Error: {e}", "optuna.log")
        continue

