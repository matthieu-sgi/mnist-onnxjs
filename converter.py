
from mnist import MnistModelCNN

import torch
import onnx

if __name__ == "__main__":
    model = MnistModelCNN()
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    model.to('cpu')
    example = torch.rand(1, 1, 28, 28)
    # traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save("mnist_cnn.pt")

    #Export to ONNX
    torch.onnx.export(model,
                      example,
                      "./web_demo/onnx_model.onnx",
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11,
                      do_constant_folding=True,
                      export_params=True,
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    onnx_model = onnx.load("./web_demo/onnx_model.onnx")
    onnx.checker.check_model(onnx_model)
    
    # Compare torch model to onnx model
    import comparator