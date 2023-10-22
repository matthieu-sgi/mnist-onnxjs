const video = document.getElementById('webcam');
const switchCamBtn = document.getElementById('switchCam');
const predictionText = document.getElementById('prediction');
let currentStream;
let model;

// Initialize the plot
const initialData = [{
    y: Array(10).fill(0),  // Assuming 10 logits for MNIST
    type: 'bar'           // Specify the type as 'bar'
}];
Plotly.newPlot('logitsGraph', initialData);

// Initialize ONNX model
async function initModel() {
    model = new onnx.InferenceSession({ backendHint: 'webgl' });
    await model.loadModel('./onnx_model.onnx');
}

// Get webcam access
async function getWebcam(streamName = 'user') {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: streamName
        }
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;

    video.onloadedmetadata = function() {
        video.play();
        inferenceLoop();
    };
}

function preprocessFrame() {
    // Get video element
    const video = document.getElementById('webcam');

    // Create a temporary canvas to capture a frame from the video
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0);

    // Create an off-screen canvas for resizing
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = 28;
    resizeCanvas.height = 28;
    const resizeCtx = resizeCanvas.getContext('2d');

    resizeCtx.drawImage(captureCanvas, video.videoWidth / 2 - 140, video.videoHeight / 2 - 140, 280, 280, 0, 0, 28, 28);

    // Get resized image data
    const resizedImageData = resizeCtx.getImageData(0, 0, 28, 28);

    // var grayscaleValue=0;
    // Convert to grayscale & normalize
    const input = new Float32Array(28 * 28);
    for(let i = 0; i < 28; i++) {
        for(let j = 0; j < 28; j++) {
            const idx = i * 28 + j;
            const r = resizedImageData.data[idx * 4];
            const g = resizedImageData.data[idx * 4 + 1];
            const b = resizedImageData.data[idx * 4 + 2];
            const grayscaleValue = ((0.299 * r + 0.587 * g + 0.114 * b)/255) > 0.3 ? 0.5 : -0.5;
            input[idx]= grayscaleValue;

            // Set the grayscaled value to the imageData for display
            resizedImageData.data[idx * 4] = (0.5+grayscaleValue)*255;
            resizedImageData.data[idx * 4 + 1] = (0.5+grayscaleValue)*255;
            resizedImageData.data[idx * 4 + 2] = (0.5+grayscaleValue)*255;
            resizedImageData.data[idx * 4 + 3] = 255; // Alpha channel
        }
    }
    // Display the grayscaled and resized image on the actual canvas in the HTML
    const displayCanvas = document.getElementById('grayscaleCanvas');
    const displayCtx = displayCanvas.getContext('2d');
    displayCtx.putImageData(resizedImageData, 0, 0);

    const output_tensor = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
    return output_tensor;
}




// Inference loop
async function inferenceLoop() {
    // const ctx = video.getContext('2d');
    const tensorInput = preprocessFrame();
    const output = await model.run([tensorInput]);
    const max_index = output.values().next().value.data.indexOf(Math.max(...output.values().next().value.data));
    predictionText.innerHTML = max_index;

    // Process logits & plot
    const logits = output.values().next().value.data;
    const updatedData = [{ y: logits , type: 'bar'}];
    Plotly.react('logitsGraph', updatedData);

    const now = Date.now();
    const fps = 1000 / (now - (window.lastInferenceTime || now));
    window.lastInferenceTime = now;
    // console.log(`FPS: ${fps.toFixed(2)}, logits: ${logits}`);

    requestAnimationFrame(inferenceLoop);
}

switchCamBtn.addEventListener('click', () => {
    const facingMode = video.srcObject.getVideoTracks()[0].getSettings().facingMode;
    getWebcam(facingMode === 'user' ? 'environment' : 'user');
});

initModel().then(() => {
    getWebcam();
});
