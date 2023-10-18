import torch
import onnx
import onnxruntime
import numpy as np
from mnist import MnistModelCNN
from torchvision import datasets, transforms
import tqdm

# Define a function to preprocess an image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)
    return image

if __name__ == "__main__":
    # Load the PyTorch model
    model = MnistModelCNN()
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    model.to('cpu')

    # Load the ONNX model
    onnx_model = onnx.load("onnx_model.onnx")
    onnx.checker.check_model(onnx_model)

    # Create an ONNX Runtime session for the ONNX model
    ort_session = onnxruntime.InferenceSession("onnx_model.onnx")

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=False)

    # Compare torch model to onnx model
    num_images_to_compare = 100
    correct_count = 0
    print('ehre')
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

        if predicted_label_torch == label.item() and predicted_label_onnx == label.item():
            correct_count += 1

    accuracy = correct_count / len(dataloader)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(onnx.checker.check_model(onnx_model))
    print(onnx.helper.printable_graph(onnx_model.graph))
