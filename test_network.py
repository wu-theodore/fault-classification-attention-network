import os
import sys
import json
import torch.nn as nn

from utils.test_utils import load_test_data, create_onnxruntime_session, run_inference, compare_pred_result, save_results, create_confusion_matrix, compute_binary_pred_metrics

def test_network(config_file, noise_var):
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load test data
    test_loader = load_test_data(config["test_data_dir"], config["model"], noise_var=noise_var)
    criterion = nn.CrossEntropyLoss()

    test_loss = []
    test_accuracy = []
    precisions = []
    recalls = []
    f1s = []
    for fold in range(config["num_folds"]):
        # Create OnnxRuntime session
        onnx_model_path = os.path.join(config["model_dir"], f"{config['model']}_{fold}" + ".onnx")
        ort_session = create_onnxruntime_session(onnx_model_path)

        # Run inference loop and compute results
        total_samples = len(test_loader)
        running_loss = 0.0
        num_correct = 0
        # For generating confusion matrix
        preds = []
        labels = []
        for sample, label in test_loader:
            result = run_inference(ort_session, sample)
            loss, pred, correct = compare_pred_result(result, label, criterion)
            running_loss += loss
            if correct:
                num_correct += 1

            preds.append(pred.item())
            labels.append(label.item())

        test_loss.append(running_loss / total_samples)
        test_accuracy.append(num_correct / total_samples)

        create_confusion_matrix(preds, labels, save_path=os.path.join(config["save_dir"], f"{config['model']}_{fold}_cm.png"), show=False)
        precision, recall, f1, support = compute_binary_pred_metrics(preds, labels)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    save_results(test_loss, test_accuracy, precisions, recalls, f1s, save_path=os.path.join(config["save_dir"], f"test_results_{config['model']}.txt"))

if __name__ == "__main__":
    assert len(sys.argv) == 2 or len(sys.argv) == 3, "Incorrect number of arguments. Expected: python test_network.py <config_file>"
    if len(sys.argv) == 3:
        noise_var = float(sys.argv[2])
    else:
        noise_var = 0

    test_network(config_file=sys.argv[1], noise_var=noise_var)