import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader

from utils.CAVSignalDataset import CAVSignalDataset
from utils.Transforms import MinMaxScale, Sample, Compose

def load_test_data(data_dir, batch_size=1, shuffle=True):
    transforms = Compose([MinMaxScale(), Sample(sample_freq=2)])
    dataset = CAVSignalDataset(data_dir, transform=transforms)
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

def compare_pred_result(output, label, criterion):
    if type(output) != torch.TensorType:
        output = torch.tensor(output, dtype=torch.float)

    label_map = {
        0: "actuator",
        1: "attack",
        2: "delay",
        3: "distracted",
        4: "drunk",
    }
    loss = criterion(output, label).item()
    pred = torch.argmax(output, dim=1)
    correct = (pred == label).item()
    print(f"Model predicted class {label_map[pred.item()]}, true class was {label_map[label.item()]}")
    return loss, correct

def save_results(loss, acc, file_path):
    with open(file_path, 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {acc}")
    

