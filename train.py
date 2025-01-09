import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch_optimizer import Lookahead
from src.data.config import dataset, data_loader, model as model_config, optimizer as optimizer_config, scheduler as scheduler_config, training

# Import custom modules
from src.models.model import KeywordSpottingModel_with_cls
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter, load_bg_noise_dataset
from src.utils.utils import set_memory_GB, print_model_size, log_to_file, plot_learning_curves
from src.utils.augmentations import add_time_shift_and_align, add_silence
from src.utils.train_utils import trainig_loop

# Load datasets
train_ds, val_ds, test_ds, silence_ds , info = load_speech_commands_dataset(reduced=False)
bg_noise_ds = load_bg_noise_dataset()

# Initialize datasets with configurations
pytorch_train_dataset = TFDatasetAdapter(train_ds, bg_noise_ds, **dataset, augmentation=[lambda x: add_time_shift_and_align(x)])
pytorch_val_dataset = TFDatasetAdapter(val_ds, None, **dataset, augmentation=None)
#print labels
      # Create DataLoaders
train_loader = DataLoader(pytorch_train_dataset, **data_loader, shuffle=True)
val_loader = DataLoader(pytorch_val_dataset, **data_loader, shuffle=False)

# Initialize model
model = KeywordSpottingModel_with_cls(**model_config).to("cuda")

# Loss function
criterion = nn.CrossEntropyLoss().to("cuda")

# Optimizer
base_optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
optimizer = Lookahead(base_optimizer, **optimizer_config['lookahead'])

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config['reduce_lr_on_plateau'])

# Training loop
num_epochs = training['num_epochs']
try:
    train_accuracies, val_accuracies, train_losses, val_losses = trainig_loop(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler)
except Exception as err:
    log_to_file(str(err))
# Plot results
plot_learning_curves(train_accuracies, val_accuracies, train_losses, val_losses, save_to_file=True)