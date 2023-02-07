import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_history(train_history, val_history, show=True, save_dir=None):
    x_train = np.arange(train_history['epoch_num'])
    x_val = np.arange(val_history['epoch_num'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_curve(axes[0], x_train, train_history["loss"], label="Train", title="Loss Curve", ylabel="Loss")
    plot_curve(axes[0], x_val, val_history["loss"], label="Validation", title="Loss Curve", ylabel="Loss")
    plot_curve(axes[1], x_train, train_history["accuracy"], label="Train", title="Accuracy Curve", ylabel="Accuracy")
    plot_curve(axes[1], x_val, val_history["accuracy"],  label="Validation", title="Accuracy Curve", ylabel="Accuracy")
    
    if save_dir:
        plt.savefig(save_dir)
    if show: 
        plt.show()
    

def plot_curve(ax, x_data, y_data, label, title, ylabel):
    ax.plot(x_data, y_data, label=label)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)


def plot_attention_weights_heatmap(device, model, data_loader, show=True, save_dir=None):
    """
    Take the first signal in data_loader and visualize attention weights.
    """
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, weights, embeddings = outputs

            sample_input = inputs[0, :].cpu().squeeze().numpy()
            sample_label = labels[0].cpu().item()
            sample_weights = torch.squeeze(weights)[0, :, :].cpu().squeeze().numpy()
            plot_heatmap(sample_input, sample_label, sample_weights)
            break

    if save_dir:
        plt.savefig(save_dir)
    if show: 
        plt.show()

def plot_heatmap(sample, label, weights):
    label_map = {
        0: "actuator",
        1: "attack",
        2: "delay",
        3: "distracted",
        4: "drunk",
    }

    seq_len = sample.shape[0]
    num_heads = weights.shape[-1] // seq_len

    x = np.arange(sample.shape[0])

    # Plot the sample
    fig, ax = plt.subplots(num_heads + 1, 1, figsize=(16, 8))
    ax[0].plot(x, sample, label=[f"{label_map[label]}_{j}" for j in range(3)])
    ax[0].legend(loc="upper right", prop={'size': 8})
    ax[0].tick_params(axis='x', labelsize=8)
    ax[0].set_xticks(range(0, seq_len, 20))

    for head in range(num_heads):
        attention_head_weights = weights[:, head*seq_len:(head + 1)*seq_len]
        weighted_sample = attention_head_weights @ sample
        im = ax[head + 1].matshow(weighted_sample.T, cmap='viridis', aspect="auto")
        ax[head + 1].set_xticks(range(0, seq_len, 20))
        ax[head + 1].xaxis.set_ticks_position("bottom")
        ax[head + 1].set_ylabel(f"Head {head + 1} of {num_heads}")
        plt.colorbar(im, ax=ax[head + 1], fraction=0.046, pad=0.01)
        

        
        
    

        
