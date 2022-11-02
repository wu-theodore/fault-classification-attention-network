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


