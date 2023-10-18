import torch
import onnx
import onnxruntime
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision import models
from mnist import transforms_train
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the ONNX model
onnx_model_path = "./web_demo/onnx_model.onnx"  # Replace with the path to your ONNX model
onnx_model = onnx.load(onnx_model_path)

# Step 2: Load the MNIST dataset and select an image
# transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = MNIST(root='./datasets', train=False, transform=transforms_train, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

while True : 
    image, label = next(iter(mnist_loader))

    # Step 3: Run inference on the image using the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_outs[0])

    ## Display the source image
    image = image.squeeze().squeeze()  # Remove batch and channel dimensions
    plt.subplot(1, 2, 1)
    plt.title("Source Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Display the model's output as a bar chart
    output = output.squeeze().detach().cpu().numpy()
    plt.subplot(1, 2, 2)
    plt.title("Model Output (Probabilities)")
    plt.bar(range(10), output)
    plt.xticks(range(10), range(10))
    plt.xlabel("Class")
    plt.ylabel("Probability")

    plt.show()


