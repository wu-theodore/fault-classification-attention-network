import sys
import numpy as np
import torch
import train_svm
import train_network

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    print(f"Training all models")
    config_names = [
        "config_mlp.json",
        "config_cnn.json",
        "config_rnn.json", 
        "config_msalstmcnn.json",
        "config_attention.json",
        "config_attention_cnn.json",
    ]

    # Train SVM separately
    print(f"Training SVM")
    train_svm.main()

    # Train other networks
    for config_file in config_names:
        print(f"Training config {config_file}")
        train_network.main(config_file=config_file)
