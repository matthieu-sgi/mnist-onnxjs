import torch
import onnx
import onnxruntime
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision import models
from mnist import transforms_train, transforms_test
import matplotlib.pyplot as plt
import numpy as np
from mnist import MnistModelCNN

torch_model = MnistModelCNN()
torch_model.load_state_dict(torch.load('best_model.pt'))

# Step 1: Load the ONNX model
onnx_model_path = "./web_demo/onnx_model.onnx"  # Replace with the path to your ONNX model
onnx_model = onnx.load(onnx_model_path)

# Step 2: Load the MNIST dataset and select an image
# transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = MNIST(root='./datasets', train=False, transform=transforms_train, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

while True : 
    image, label = next(iter(mnist_loader))
    print(image.mean(), image.std(), image.shape, image.max(), image.min())

    # Step 3: Run inference on the image using the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_outs[0])

    # Step 4: Crun torch model
    torch_out = torch_model(image)


    ## Display the source image
    image = image.squeeze().squeeze()  # Remove batch and channel dimensions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Source Image: ({image.shape[0]}, {image.shape[1]})| label: {label.item()}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Display the model's output as a bar chart
    output = output.squeeze().detach().cpu().numpy()
    plt.subplot(1, 3, 2)
    plt.title("Onnx Model Output (Probabilities)")
    plt.bar(range(10), output)
    plt.xticks(range(10), range(10))
    plt.xlabel("Class")
    plt.ylabel("Probability")

    # Display the torch model's output as a bar chart
    torch_output = torch_out.squeeze().detach().cpu().numpy()
    plt.subplot(1, 3, 3)
    plt.title("Torch Model Output (Probabilities)")
    plt.bar(range(10), torch_output)
    plt.xticks(range(10), range(10))
    plt.xlabel("Class")
    plt.ylabel("Probability")


    plt.show()


