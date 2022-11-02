import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.test_utils import load_test_data
from utils.train_utils import load_data

def visualize_signal(data_loader, title=None, show=True, save_dir=None):
    label_map = {
        0: "actuator",
        1: "attack",
        2: "delay",
        3: "distracted",
        4: "drunk",
    }
    fig, axes = plt.subplots(5, 1, figsize=(16, 8))
            
    fig.suptitle(title, fontsize=16)

    for i in range(len(axes)):
        for sample, label in data_loader:
            sample, label = sample.squeeze().numpy(), label.item()
            if label == i:
                x = np.arange(sample.shape[0])
                axes[i].plot(x, sample, label=[f"{label_map[i]}_{j}" for j in range(3)])
                axes[i].legend(loc="upper right", prop={'size': 8})
                axes[i].tick_params(axis='x', labelsize=8)

                break
    
    if save_dir:
        plt.savefig(save_dir)
    if show:
        plt.show()

def visualize_data():
    with open("config.json", 'r') as f:
        config = json.load(f)

    train_loader, _ = load_data(config["train_data_dir"], train_split=0.99, batch_size=1, shuffle=True)
    test_loader = load_test_data(config["test_data_dir"])

    visualize_signal(train_loader, title="Training Data Signal", save_dir=os.path.join(config["save_dir"], "train_data_distribution"))
    visualize_signal(test_loader, title="Test Data Signal", save_dir=os.path.join(config["save_dir"], "test_data_distribution"))


if __name__ == "__main__":
    visualize_data()