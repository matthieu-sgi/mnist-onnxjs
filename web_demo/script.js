const CANVAS_SIZE = 28;
let isPlotInitialized = false;


const webcamElement = document.getElementById('webcam');
const switchCameraButton = document.getElementById('switch-camera');
const sess = new onnx.InferenceSession();

// Store all available video devices
let videoDevices = [];
let currentDeviceIndex = 0;

// Load the ONNX model
const loadingModelPromise = sess.loadModel('./onnx_model.onnx');

// Initialize the webcam stream
navigator.mediaDevices.enumerateDevices().then(devices => {
    videoDevices = devices.filter(device => device.kind === 'videoinput');
    startWebcamStream(videoDevices[currentDeviceIndex].deviceId);
});

switchCameraButton.addEventListener('click', () => {
    currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
    startWebcamStream(videoDevices[currentDeviceIndex].deviceId);
});

function startWebcamStream(deviceId) {
    navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId }
    }).then(stream => {
        webcamElement.srcObject = stream;
        webcamElement.play();
    }).catch(error => {
        console.error('Error accessing the webcam:', error);
    });
}

function reshapeFloat32ArrayToUint8ClampedArray(inputArray) {
    const outputArray = new Uint8ClampedArray(inputArray.length);
    
    for (let i = 0; i < inputArray.length; i++) {
        // Convert the float value to the 0-255 range
        const clampedValue = Math.min(255, Math.max(0, Math.round(inputArray[i] * 255)));
        outputArray[i] = clampedValue;
    }

    return outputArray;
}

function preprocessFrame(videoElement) {
    const canvas = document.createElement('canvas');
    const imga = document.getElementById('transformed');
    canvas.width = CANVAS_SIZE; // 28
    canvas.height = CANVAS_SIZE; // 28
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const inputData = new Float32Array(1 * 28 * 28);
    console.log(imageData.data.length);
    for (let i = 0; i < imageData.data.length; i += 4) {
        inputData[i / 4] = ((imageData.data[i] / 255.0) - 0.1307)/0.3081;
    }

    // Display input data on the webpage
    const ctx_2 = imga.getContext('2d');
    // ctx_2.(imageData, 0, 0);
    // InputData is a float32 Array and has to be reshaped to a Uint8ClampedArray
    const reshaped_inputImage = reshapeFloat32ArrayToUint8ClampedArray(inputData);
    ctx_2.putImageData(reshaped_inputImage, 0, 0);

    // imga.putImageData(imageData, 0, 0);
    // document.body.appendChild(imga);
    // const ctx_2 = canvas_2.getContext('2d');

    console.log(inputData)
    return new onnx.Tensor(inputData, 'float32', [1, 1, 28, 28]);
}

async function processWebcamFrame() {
    if (webcamElement.readyState === webcamElement.HAVE_ENOUGH_DATA) {
        const input = preprocessFrame(webcamElement);
        const startTime = performance.now();
        const outputMap = await sess.run([input]);
        const endTime = performance.now();

        const elapsedTime = endTime - startTime;
        const fps = 1000 / elapsedTime;
        console.log(`Frame processed in ${elapsedTime.toFixed(2)} ms (${fps.toFixed(2)} FPS)`);

        updateHistogram(outputMap.values().next().value.data);
    }
    requestAnimationFrame(processWebcamFrame);
}


function updateHistogram(predictions) {
    const trace = {
        x: Array.from({ length: 10 }, (_, i) => i),
        y: predictions,
        type: 'bar'
    };
    
    if (!isPlotInitialized) {
        Plotly.newPlot('plotly-histogram', [trace]);
        isPlotInitialized = true;
    } else {
        Plotly.update('plotly-histogram', { y: [predictions] });
    }
}

loadingModelPromise.then(() => {
    processWebcamFrame();
});