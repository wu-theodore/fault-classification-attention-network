import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader

from utils.CAVSignalDataset import CAVSignalDataset
from utils.Transforms import MinMaxScale, Sample, Compose, ExtractTimeDomainFeatures, GaussianNoise
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    
    if model == "cnn" or model == "msalstm-cnn":
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

def save_results(loss, acc, save_path):
    with open(save_path, 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {acc}")
    
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
        plt.savefig(save_path, format='eps', dpi=600)
    if show:
        plt.show()
