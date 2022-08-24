import os
import json
import torch.nn as nn

from utils.test_utils import load_test_data, create_onnxruntime_session, run_inference, compare_pred_result, save_results

def test_network():
    # Load config
    with open("config.json", 'r') as f:
        config = json.load(f)

    # Load test data
    test_loader = load_test_data(config["test_data_dir"])
    criterion = nn.CrossEntropyLoss()
    # Create OnnxRuntime session
    onnx_model_path = os.path.join(config["model_dir"], config["model"] + ".onnx")
    ort_session = create_onnxruntime_session(onnx_model_path)

    # Run inference loop and compute results
    total_samples = len(test_loader)
    running_loss = 0.0
    num_correct = 0
    for sample, label in test_loader:
        result = run_inference(ort_session, sample)
        loss, correct = compare_pred_result(result, label, criterion)
        running_loss += loss
        if correct:
            num_correct += 1
    test_loss = running_loss / total_samples
    test_accuracy = num_correct / total_samples

    save_path = os.path.join(config["save_dir"], "test_results.txt")
    save_results(test_loss, test_accuracy, save_path)

if __name__ == "__main__":
    test_network()