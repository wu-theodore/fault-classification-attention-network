import sys
from test_network import test_network
from test_svm import test_svm

if __name__ == "__main__":
    assert len(sys.argv) == 2
    noise_var = float(sys.argv[-1])

    print(f"Running test with noise variance level {noise_var}")
    config_names = [
        "config_mlp.json",
        "config_cnn.json",
        "config_rnn.json", 
        "config_msalstmcnn.json",
        "config_attention.json",
    ]

    # Test SVM separately
    test_svm(noise_var)

    # Test other networks
    for config_file in config_names:
        print(f"Testing config {config_file}")
        test_network(config_file, noise_var)
