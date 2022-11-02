import os
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from utils.EarlyStopping import EarlyStopping
from utils.plot_utils import plot_history
from utils.save_utils import save_metrics_history, save_model
from utils.train_utils import *


def train_one_epoch(device, model, data_loader, criterion, optimizer):
    running_loss = 0.0
    running_batch_accuracy = 0.0
    batch_num = 0

    model.train()
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_accuracy = compute_batch_accuracy(outputs.detach(), labels)
        batch_loss = loss.item()
        
        running_batch_accuracy += batch_accuracy
        running_loss += batch_loss
        batch_num += 1

    train_loss = running_loss / batch_num
    train_accuracy = running_batch_accuracy / batch_num
    torch.cuda.empty_cache()
    return train_loss, train_accuracy


def validate(device, model, data_loader, criterion):
    running_loss = 0.0
    running_batch_accuracy = 0.0
    batch_num = 0
    
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            batch_accuracy = compute_batch_accuracy(outputs.detach(), labels)
            batch_loss = loss.item()
            
            running_batch_accuracy += batch_accuracy
            running_loss += batch_loss
            batch_num += 1

    val_loss = running_loss / batch_num
    val_accuracy = running_batch_accuracy / batch_num
    return val_loss, val_accuracy

def train(config, device, model, data, criterion, optimizer, print_every=100, save_every=10, use_checkpoint=False, use_wandb=False, early_stop=False):
    if early_stop:
        early_stopping = EarlyStopping()

    # Load checkpoint if exists and desired.
    checkpoint_path = "checkpoint.pt"
    start_epoch = 0

    if use_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    train_history = dict()
    train_history['loss'] = []
    train_history['accuracy'] = []
    train_history['epoch_num'] = 0

    val_history = dict()
    val_history['loss'] = []
    val_history['accuracy'] = []
    val_history['epoch_num'] = 0

    train_loader, val_loader = data

    for epoch in range(start_epoch, config["epochs"]):
        # Run training
        train_loss, train_accuracy = train_one_epoch(device, model, train_loader, criterion, optimizer)

        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_accuracy)
        train_history['epoch_num'] += 1

        # Run validation    
        val_loss, val_accuracy = validate(device, model, val_loader, criterion)
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_accuracy)
        val_history['epoch_num'] += 1

        # Log to wandb if using
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "epoch": epoch
            })

        if early_stop:
            stop = early_stopping(val_loss, model)
            if stop:
                break

        # Display stats
        if (epoch + 1) % print_every == 0:
            print("-------------------------------")
            print(f'[Epoch {epoch + 1}] Training loss: {train_loss:.3f}')
            print(f'[Epoch {epoch + 1}] Training accuracy: {train_accuracy:.3f}')
            print(f'[Epoch {epoch + 1}] Validation loss: {val_loss:.3f}')
            print(f'[Epoch {epoch + 1}] Validation accuracy: {val_accuracy:.3f}') 

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy
            }, checkpoint_path)

    return train_history, val_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch running on {device}")
    
    with open("config.json", 'r') as f:
        config = json.load(f)

    # Instantiate DataLoaders
    train_loader, val_loader = load_data(config["train_data_dir"], config["train_val_split"], batch_size=config["batch_size"])

    # Instantiate Networks
    model = load_model(config, device)

    # Define optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train model
    train_history, val_history = train(config, device, model, (train_loader, val_loader),
        criterion, optimizer, early_stop=False, print_every=config["verbosity"])

    # Plot/Save results
    plot_history(train_history, val_history, save_dir=os.path.join(config["save_dir"], f"{config['model']}_training_curves.png"))
    save_metrics_history(train_history, save_path=os.path.join(config["save_dir"], f"{config['model']}_train_history")) 
    save_metrics_history(val_history, save_path=os.path.join(config["save_dir"], f"{config['model']}_val_history")) 

    # Save trained model
    sample_input = torch.randn(size=(config["batch_size"], 500, config["state_size"])).to(device)
    save_model(model, sample_input, save_path=os.path.join(config["model_dir"], config["model"]))

if __name__ == "__main__":
    torch.manual_seed(0)
    main()


