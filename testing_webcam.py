import torch
import onnx
import onnxruntime
import torchvision.transforms as transforms
import cv2
import numpy as np
from mnist import MnistModelCNN

torch_model = MnistModelCNN()
torch_model.load_state_dict(torch.load('best_model.pt'))

# Step 1: Load the ONNX model
onnx_model_path = "./web_demo/onnx_model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Step 2: Set up webcam capture
cap = cv2.VideoCapture(0)


def draw_bar_chart(frame, logits, title="Logits"):
    height, width, _ = frame.shape
    bar_width = width // 10
    max_val = max(logits)
    min_val = min(logits)
    span = max_val - min_val
    for i, value in enumerate(logits):
        normalized_value = (value - min_val) / span
        draw_height = int(normalized_value * height)
        start_point = (i * bar_width, height - draw_height)
        end_point = ((i + 1) * bar_width, height)
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), -1)
        cv2.putText(frame, f"{i}", (i * bar_width + bar_width // 4, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to match the expected input size (28x28 for MNIST) and normalize
    resized_frame = cv2.resize(gray_frame, (28, 28))
    processed_image = (transforms.ToTensor()(resized_frame).unsqueeze(0)>0.3).type(torch.float32) - 0.5
    print(processed_image.mean(), processed_image.std(), processed_image.shape, processed_image.max(), processed_image.min())

    # print(processed_image.mean(), processed_image.std(), processed_image.shape, processed_image.max(), processed_image.min())

    # print(processed_image)

    # Step 3: Run inference on the image using the ONNX model
    output = torch_model(processed_image)
    output = output.squeeze().detach().cpu().numpy()
    print(output.argmax())
    # ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: processed_image.detach().cpu().numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)
    # output = ort_outs[0].squeeze()

    # Display the model's output as a bar chart using OpenCV
    output_frame = np.zeros_like(frame)
    draw_bar_chart(output_frame, output)

    cv2.imshow("Model Output", output_frame)

    display_image = processed_image.squeeze().detach().cpu().numpy()
    # Display the source image using OpenCV
    cv2.imshow("Source Image", display_image)

    # # Display the model's output using OpenCV (for simplicity, only displaying the Onnx model's output as text)
    # predicted_label = np.argmax(output.squeeze().detach().cpu().numpy())
    # label_text = f"Predicted label: {predicted_label}"
    # cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow("Prediction", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
