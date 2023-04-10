import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader

from utils.CAVSignalDataset import CAVSignalDataset
from utils.Transforms import MinMaxScale, Sample, Compose, ExtractTimeDomainFeatures, GaussianNoise
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

label_map = {
        0: "Actuator",
        1: "FDI",
        2: "DoS",
        3: "Distracted",
        4: "Drunk",
    }

def load_test_data(data_dir, model, noise_var=0, batch_size=1, shuffle=True):
    if model == "mlp":
        transforms = [ExtractTimeDomainFeatures(), MinMaxScale(1, 0)]
    else:
        transforms = [MinMaxScale()] 
    if noise_var:
        transforms.append(GaussianNoise(variance=noise_var))
    transforms = Compose(transforms)
    
    if model == "cnn" or model == "msalstm-cnn" or model == "attention-cnn":
        channel_first = True
    else:
        channel_first = False

    dataset = CAVSignalDataset(data_dir, transform=transforms, channel_first=channel_first) 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def create_onnxruntime_session(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    return ort_session

def run_inference(ort_session, sample):
    def to_numpy(tensor):
        return tensor.detach().cpu().float().numpy() if tensor.requires_grad else tensor.cpu().float().numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sample)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def compare_pred_result(output, label, criterion, print_result=False):
    if type(output) != torch.TensorType:
        output = torch.tensor(output, dtype=torch.float)

    loss = criterion(output, label).item()
    pred = torch.argmax(output, dim=1)
    correct = (pred == label).item()
    if print_result:
        print(f"Model predicted class {label_map[pred.item()]}, true class was {label_map[label.item()]}")
    return loss, pred, correct

def compute_binary_pred_metrics(pred, true):
    return precision_recall_fscore_support(true, pred, average=None, labels=list(label_map.keys()))

def save_results(loss, acc, precision, recall, f1, save_path):
    def compute_ci(data, alpha=0.95):
        ci_low, ci_high = stats.t.interval(alpha, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
        return (ci_high - ci_low) / 2

    # Precision, recall, and f1 currently list of arrays, convert to 2d array
    precision = np.stack(precision, axis=0)
    recall = np.stack(recall, axis=0)
    f1 = np.stack(f1, axis=0)

    with open(save_path, 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {np.mean(acc)} +- {compute_ci(acc)}, \t\t {acc}\n")
        for i in range(len(label_map)):
            f.write("\n")
            f.write(f"Precision of class {i}: {np.mean(precision[:, i])} +- {compute_ci(precision[:, i])}, \t\t {precision[:, i]}\n")
            f.write(f"Recall of class {i}: {np.mean(recall[:, i])} +- {compute_ci(recall[:, i])}, \t\t {recall[:, i]}\n")
            f.write(f"F1 of class {i}: {np.mean(f1[:, i])} +- {compute_ci(f1[:, i])}, \t\t {f1[:, i]}\n")
    
def create_confusion_matrix(output, label, save_path=None, show=False):
    cm = confusion_matrix(label, output)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=label_map.values())
    cm_display.plot(colorbar=False, cmap="Blues")
    font = {'fontname':'Times New Roman', 'fontsize':16}
    plt.xlabel("Predicted Label", **font)
    plt.xticks(**font)
    plt.ylabel("True Label", **font)
    plt.yticks(**font)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
