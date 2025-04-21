import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.config import dataset, data_loader, model as model_config, optimizer as optimizer_config, scheduler as scheduler_config
from src.data.data_loader import load_speech_commands_dataset, TFDatasetAdapter
from src.models.model import KeywordSpottingModel_with_cls
from src.utils.utils import log_to_file
from src.utils.train_utils import trainig_loop

def train_model(num_epochs=100):
    train_ds, val_ds, test_ds, silence_ds, info = load_speech_commands_dataset(reduced=False)
    
    pytorch_train_dataset = TFDatasetAdapter(train_ds, **dataset)
    pytorch_val_dataset = TFDatasetAdapter(val_ds, **dataset)
    
    train_loader = DataLoader(pytorch_train_dataset, **data_loader, shuffle=True)
    val_loader = DataLoader(pytorch_val_dataset, **data_loader, shuffle=False)
    
    model = KeywordSpottingModel_with_cls(**model_config).to("cuda")
    criterion = nn.CrossEntropyLoss().to("cuda")
    
    optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config['reduce_lr_on_plateau'])
    
    log_to_file("Starting training...")
    
    train_accuracies, val_accuracies, train_losses, val_losses = trainig_loop(
        model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler
    )
    
    log_to_file("Training completed.")
    
    return train_accuracies, val_accuracies, train_losses, val_losses

if __name__ == "__main__":
    train_model()