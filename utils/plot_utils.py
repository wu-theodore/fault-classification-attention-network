import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_history(train_history_list, val_history_list, show=True, save_dir=None):
    for i, (train_history, val_history) in enumerate(zip(train_history_list, val_history_list)):
        if i == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if type(train_history["epoch_num"]) == int:
            x_train = np.arange(train_history['epoch_num'])
        else:
            x_train = np.array(train_history["epoch_num"])
        if type(val_history["epoch_num"]) == int:
            x_val = np.arange(val_history['epoch_num'])
        else:
            x_val = np.array(val_history["epoch_num"])

        plot_curve(axes[0], x_train, train_history["loss"], label=f"Train_{i}", title="Loss Curve", ylabel="Loss")
        plot_curve(axes[0], x_val, val_history["loss"], label=f"Validation_{i}", title="Loss Curve", ylabel="Loss")
        plot_curve(axes[1], x_train, train_history["accuracy"], label=f"Train_{i}", title="Accuracy Curve", ylabel="Accuracy")
        plot_curve(axes[1], x_val, val_history["accuracy"],  label=f"Validation_{i}", title="Accuracy Curve", ylabel="Accuracy")
    
    if save_dir:
        plt.savefig(save_dir)
    if show: 
        plt.show()

    plt.close()
    

def plot_curve(ax, x_data, y_data, label, title, ylabel):
    ax.plot(x_data, y_data, label=label)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)


def plot_attention_weights_heatmap(device, model, data_loader, show=False, save_dir=None):
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
            sample_weights = weights[-1, 0, :, :].cpu().squeeze().numpy()

            plot_heatmap(sample_input, sample_label, sample_weights, save_dir, show)
            break


def plot_heatmap(sample, label, weights, save_dir=None, show=None):
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

    for head in range(num_heads):
        # Plot the sample
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8), constrained_layout=True)

        ax[0].plot(x, sample, label=[f"{label_map[label]}_{j}" for j in range(3)])
        ax[0].legend(loc="upper right", prop={'size': 8})
        ax[0].tick_params(axis='x', labelsize=8)
        ax[0].set_xticks(range(0, seq_len, 20))

        attention_head_weights = weights[:1, head*seq_len:(head + 1)*seq_len]
        im = ax[1].matshow(attention_head_weights, cmap='viridis', aspect="auto")
        ax[1].set_xticks(range(0, seq_len, 20))
        ax[1].xaxis.set_ticks_position("bottom")
        ax[1].set_title(f"Head {head + 1} of {num_heads}")
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.01)

        if save_dir:
            plt.savefig(save_dir + f"_head{head + 1}_attention_heatmap.png")
        if show:
            plt.show()
    plt.close()
