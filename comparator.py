import torch
import onnx
import onnxruntime
import numpy as np
from mnist import MnistModelCNN, transforms_test
from torchvision import datasets, transforms
import tqdm

# Define a function to preprocess an image
def preprocess_image(image):
    image = transforms_test(image)
    return image

# Load the PyTorch model
model = MnistModelCNN()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
model.to('cpu')

# Load the ONNX model
onnx_model = onnx.load("./web_demo/onnx_model.onnx")
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session for the ONNX model
ort_session = onnxruntime.InferenceSession("./web_demo/onnx_model.onnx")

# Load the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset = datasets.MNIST(root='./datasets', train=False, transform=transforms_test, download=True)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=False)

# Compare torch model to onnx model
num_images_to_compare = 100
correct_count = 0
actual_correct = 0
for i, (image, label) in enumerate(tqdm.tqdm(dataloader)):
    
    # if i >= num_images_to_compare:
    #     break

    # PyTorch inference
    with torch.no_grad():
        output_torch = model(image)

    # ONNX inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    image_np = image.numpy()
    output_onnx = ort_session.run([output_name], {input_name: image_np})[0]

    # Compare the results
    predicted_label_torch = torch.argmax(output_torch, dim=1).item()
    predicted_label_onnx = np.argmax(output_onnx)

    if predicted_label_torch ==  predicted_label_onnx :
        correct_count += 1
    if predicted_label_torch == label.item():
        actual_correct += 1
    

accuracy = correct_count / len(dataloader)
actual_accuracy = actual_correct / len(dataloader)
print(f"Conversion Accuracy: {accuracy * 100:.2f}%")
print(f"True Accuracy: {actual_accuracy * 100:.2f}%")
# print()
# print(onnx.helper.printable_graph(onnx_model.graph))
