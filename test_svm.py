import os
import sys
import json
import pickle

from utils.test_utils import load_test_data, compare_pred_result, save_results, create_confusion_matrix

def test_svm(noise_var):
    # Load config
    with open("config_svm.json", 'r') as f:
        config = json.load(f)

    # Load test data
    test_loader = load_test_data(config["test_data_dir"], config["model"], noise_var=noise_var)
    test_loss = []
    test_accuracy = []
    for fold in range(config["num_folds"]):
        # Load model
        with open(os.path.join(config["model_dir"], f"{config['model']}_{fold}" + ".pickle"), "rb") as f:
            clf = pickle.load(f)

        # Run inference loop and compute results
        total_samples = len(test_loader)
        running_loss = 0.0
        num_correct = 0
        # For generating confusion matrix
        preds = []
        labels = []
        for sample, label in test_loader:
            sample = sample.numpy()
            sample = sample.reshape(sample.shape[0], -1)
            label = label.numpy()
            
            pred = clf.predict(sample)

            if pred[0] == label[0]:
                num_correct += 1

            preds.append(pred[0])
            labels.append(label[0])

        test_loss.append(running_loss / total_samples)
        test_accuracy.append(num_correct / total_samples)

        create_confusion_matrix(preds, labels, save_path=os.path.join(config["save_dir"], f"{config['model']}_{fold}_cm.eps"), show=False)

    save_results("N/A", test_accuracy, save_path=os.path.join(config["save_dir"], f"test_results_{config['model']}.txt"))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        noise_var = float(sys.argv[1])
    else:
        noise_var = 0
    test_svm(noise_var)