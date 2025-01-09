import torch
from src.utils.utils import log_to_file, EarlyStopping
from tqdm import tqdm
import matplotlib.pyplot as plt

def trainig_loop(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler,save_best_model=True,device="cpu"):
    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=5 , min_delta=0.00001)

    # Training loop
    num_epochs = num_epochs

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Log new training session
    log_to_file("\n\nNew training session\n\n")
    # Log the model architecture
    log_to_file(str(model))


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for audio, labels in (train_loader):
            audio, labels = audio.to('cuda'), labels.to('cuda')

            # Forward pass
            outputs = model(audio)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}%')

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        # Log training metrics
        log_to_file(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # # Step the scheduler
        # scheduler.step()

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to("cuda"), labels.to("cuda")
                outputs = model(audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)
        print(f'Validation Loss: {val_loss_avg}, Validation Accuracy: {val_accuracy}%')
        # Log validation metrics
        log_to_file(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss_avg)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss_avg)
        print(f'Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()}')
        log_to_file(f'Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()}')
        
        
        # Check early stopping condition
        if early_stopping.step(val_loss_avg):
            print(f"Stopping training at epoch {epoch+1} due to early stopping")
            break

        # Save best model
        
        if save_best_model and val_loss_avg == min(val_losses):
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved")


    log_to_file("Training complete.")

    return train_accuracies, val_accuracies, train_losses, val_losses


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for audio, labels in tqdm(train_loader):
        audio, labels = audio.to("cuda"), labels.to("cuda")

        # Forward pass
        outputs = model(audio)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)
    print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}%')

    return train_accuracy, train_loss


def learning_rate_finder_plot(model, train_loader, criterion, optimizer):

    max_epochs = 10

    lr_finder = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10**epoch)
    lrs = []
    losses = []

    for epoch in range(max_epochs):
        # Train for one epoch
        _, train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Record the learning rate and the corresponding loss
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(train_loss)
        
        # Step the learning rate
        lr_finder.step()

    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.show()