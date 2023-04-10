import sys
import numpy as np
import torch
from test_network import test_network
from test_svm import test_svm

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    assert len(sys.argv) == 2
    noise_var = float(sys.argv[-1])

    print(f"Running test with noise variance level {noise_var}")
    config_names = [
        "config_mlp.json",
        "config_cnn.json",
        "config_rnn.json", 
        "config_msalstmcnn.json",
        "config_attention.json",
        "config_attention_cnn.json",
    ]

    # Test SVM separately
    print(f"Testing SVM")
    test_svm(noise_var)

    # Test other networks
    for config_file in config_names:
        print(f"Testing config {config_file}")
        test_network(config_file, noise_var)
