def set_memory_GB(gb):
    import torch
    import gc

    # Set the maximum memory allocated for the GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gb / torch.cuda.get_device_properties(0).total_memory)

    # Clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1e6:.2f} million parameters")

def log_to_file(message, filename='training_log.txt'):
    with open(filename, 'a') as f:
        f.write(message + '\n')

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()