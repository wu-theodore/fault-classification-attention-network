import torch
import onnx
import onnxruntime

import numpy as np
import pandas as pd

def save_metrics_history(history, save_path):
    if save_path[-4:] != ".csv":
        save_path += ".csv"
    history["epoch_num"] = [i for i in range(history["epoch_num"])]
    metrics_df = pd.DataFrame.from_dict(history)
    metrics_df = metrics_df[["epoch_num", "loss", "accuracy"]]
    metrics_df.to_csv(save_path, index=False)


def save_model(model, sample_input, save_path):
    if save_path[-5:] != ".onnx":
        save_path += ".onnx"

    model.eval()
    print("\n------------------------\n" + f"Saving model...\n")
    torch.onnx.export(
        model, 
        sample_input, 
        save_path, 
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                    'output': {0: 'batch_size'}},
        verbose=True
    )

    # Verify onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = onnxruntime.InferenceSession(save_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sample_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    torch_out = model(sample_input)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

