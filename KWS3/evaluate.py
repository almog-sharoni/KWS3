import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.data.data_loader import TFDatasetAdapter, load_speech_commands_dataset, load_bg_noise_dataset
from src.models.model import KeywordSpottingModel_with_cls
from src.utils.utils import log_to_file

def evaluate_model(model, test_loader, device):
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
    
    return 100 * test_accuracy / test_total

def main():
    # Load datasets
    train_ds, val_ds, test_ds, silence_ds, info = load_speech_commands_dataset(reduced=False)
    bg_noise_ds = load_bg_noise_dataset()
    
    # Initialize model
    model = KeywordSpottingModel_with_cls().to("cuda")
    
    # Create test dataset and DataLoader
    test_dataset = TFDatasetAdapter(test_ds, bg_noise_ds)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate model
    test_accuracy_percentage = evaluate_model(model, test_loader, "cuda")
    
    # Log test accuracy
    log_to_file(f"Test accuracy: {test_accuracy_percentage:.2f}%")

if __name__ == "__main__":
    main()