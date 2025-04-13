import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from './knear.js';


const KNOWN_LABELS = ["peace", "wave", "fist", "thumbs_up", "snake"];
const JSON_FILENAME = 'posedata.json';
const K_VALUE = 3;
const REQUIRED_POINTS = 42;
const TEST_SPLIT_RATIO = 0.2;


const statusElement = document.getElementById("status");
const webcamButton = document.getElementById("webcamButton");
const classifyButton = document.getElementById("classifyButton");
const saveButtons = document.querySelectorAll(".saveButton");
const trainButton = document.getElementById("trainButton");
const saveJsonButton = document.getElementById("saveJsonButton");
const predictionElement = document.getElementById("predictionResult");
const trainStatusElement = document.getElementById("trainStatus");
const accuracyStatusElement = document.getElementById("accuracyStatus");
const countTotalElement = document.getElementById("count-total");

const countElements = {};
KNOWN_LABELS.forEach(label => {
    countElements[label] = document.getElementById(`count-${label}`);
});



const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let results = undefined;

let machine = new kNear(K_VALUE);
let collectedData = [];
let trainingSet = [];
let testSet = [];
let isTrained = false;




function shuffleArray(array) {
    let currentIndex = array.length, randomIndex;

    while (currentIndex > 0) {

        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }
    return array;
}


function splitData(allData, testRatio) {
    const shuffledData = shuffleArray([...allData]);
    const splitIndex = Math.floor(shuffledData.length * (1 - testRatio));
    const train = shuffledData.slice(0, splitIndex);
    const test = shuffledData.slice(splitIndex);
    console.log(`${train.length} training, ${test.length} testing.`);
    return { trainingSet: train, testSet: test };
}



function flattenLandmarks(landmarks) {
    let flatArray = [];
    if (landmarks) {
        for (const landmark of landmarks) {
            flatArray.push(landmark.x);
            flatArray.push(landmark.y);
        }
    }
    return flatArray;
}

function updateCounts() {
    const counts = {};
    KNOWN_LABELS.forEach(label => counts[label] = 0);
    let total = 0;

    for (const item of collectedData) {
        if (counts.hasOwnProperty(item.label)) {
            counts[item.label]++;
        }
        total++;
    }

    for (const label in counts) {
        if (countElements[label]) {
            countElements[label].textContent = counts[label];
        }
    }
    countTotalElement.textContent = total;


    trainButton.disabled = collectedData.length < 5;
    saveJsonButton.disabled = collectedData.length === 0;
}


function saveCurrentPose(label) {
    if (!webcamRunning) {
        alert("Please start the webcam first.");
        return;
    }
    if (!results || !results.landmarks || results.landmarks.length === 0) {
        alert("No hand detected.");
        return;
    }
    const currentPose = flattenLandmarks(results.landmarks[0]);

    collectedData.push({ points: currentPose, label: label });
    console.log(`Added pose for: ${label}. Total samples: ${collectedData.length}`);
    updateCounts();


    isTrained = false;
    trainStatusElement.textContent = "Model needs training.";
    accuracyStatusElement.textContent = "Accuracy: N/A";
    classifyButton.disabled = true;
    predictionElement.textContent = "Prediction: ?";
}


function trainModel() {
    if (collectedData.length < 5) {
        alert(`Ensure you have at least 5 data points to train and test the model. u currently have ${collectedData.length}. zzzz`);
        return;
    }

    trainStatusElement.textContent = "Splitting data & Training...";
    console.log(`Starting training process with ${collectedData.length} total samples.`);


    const { trainingSet: currentTrainingSet, testSet: currentTestSet } = splitData(collectedData, TEST_SPLIT_RATIO);
    trainingSet = currentTrainingSet;
    testSet = currentTestSet;


    machine = new kNear(K_VALUE);
    let trainedCount = 0;


    console.log(`Training KNN with ${trainingSet.length} samples.`);
    for (let item of trainingSet) {
        if (item.points && item.label && Array.isArray(item.points) && item.points.length === REQUIRED_POINTS) {
            try {
                machine.learn(item.points, item.label);
                trainedCount++;
            } catch(e) {
                console.error("Error:", item, e);
            }
        }
    }


    if (trainedCount > 0) {
        isTrained = true;
        trainStatusElement.textContent = `Model Trained (${trainedCount} samples).`;
        console.log(`KNN Training complete with ${trainedCount} valid samples.`);
        if (webcamRunning) {
            classifyButton.disabled = false;
        }

        calculateAccuracy();
    }
}


function calculateAccuracy() {

    console.log(`Calculating accuracy on ${testSet.length} test samples.`);
    let correctPredictions = 0;
    const totalTestPoses = testSet.length;

    for (const item of testSet) {
        if (item.points && item.label && Array.isArray(item.points) && item.points.length === REQUIRED_POINTS) {
            const testPose = item.points;
            const trueLabel = item.label;
            try {
                const prediction = machine.classify(testPose);
                console.log(`  Test Sample (${trueLabel}): Prediction = ${prediction}`);
                if (prediction === trueLabel) {
                    correctPredictions++;
                } else if (prediction === undefined){
                    console.warn(`    -> Prediction was undefined for test sample:`, item);
                }
            } catch (error) {
                console.error(`Error classifying test sample (${trueLabel}):`, error, item);
            }
        }
    }

    if (totalTestPoses > 0) {
        const accuracy = (correctPredictions / totalTestPoses) * 100;
        console.log(`Accuracy Calculation Complete: ${correctPredictions} correct out of ${totalTestPoses} (${accuracy.toFixed(1)}%)`);
        accuracyStatusElement.textContent = `Accuracy: ${accuracy.toFixed(1)}% (${correctPredictions}/${totalTestPoses})`;
    }
}



function classifyCurrentPose() {
    if (!webcamRunning) {
        predictionElement.textContent = "Prediction: Start webcam first.";
        return;
    }
    if (!isTrained) {
        predictionElement.textContent = "Prediction: Model not trained.";
        return;
    }
    if (!results || !results.landmarks || results.landmarks.length === 0) {
        predictionElement.textContent = "Prediction: No hand detected";
        return;
    }

    const currentPose = flattenLandmarks(results.landmarks[0]);

    if (currentPose.length !== REQUIRED_POINTS) {
        console.error(`Unexpected number of classification points: ${currentPose.length}. Expected ${REQUIRED_POINTS}.`);
        predictionElement.textContent = "Prediction: Error processing data";
        return;
    }

    try {
        let prediction = machine.classify(currentPose);
        console.log(`KNN Classification Result: ${prediction}`);
        predictionElement.textContent = `Prediction: ${prediction || 'Unknown'}`;
        if (!prediction) {
            console.warn("KNN classify returned undefined. Check data/K value.");
        }
    } catch (error) {
        console.error(error);
        predictionElement.textContent = `Prediction: Error (See console)`;
    }
}




async function loadInitialData() {
    try {
        const response = await fetch(JSON_FILENAME);
        if (!response.ok) {
            if (response.status === 404) {
                collectedData = [];
            } else {

                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } else {
            const data = await response.json();
            if (Array.isArray(data)) {

                collectedData = data.filter(item => item && Array.isArray(item.points) && item.points.length === REQUIRED_POINTS && typeof item.label === 'string');

            } else {

                collectedData = [];
            }
        }
    } catch (error) {

        console.error("An error occurred during initial data loading:", error);
        collectedData = [];
    } finally {
        updateCounts();

        if (collectedData.length >= 5) {
            trainModel();
        } else {

            trainStatusElement.textContent = "Not enough data to train automatically.";
            accuracyStatusElement.textContent = "Accuracy: N/A";
            isTrained = false;
            classifyButton.disabled = true;
        }
    }
}


function saveDataToJsonFile() {
    if (collectedData.length === 0) {
        alert("No data to save!");
        return;
    }

    try {
        const jsonString = JSON.stringify(collectedData, null, 2);
        const blob = new Blob([jsonString], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = JSON_FILENAME;
        document.body.appendChild(a);
        a.click();

        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error(error);
    }
}




const createHandLandmarker = async () => {
    statusElement.textContent = "Loading Hand Landmark Model...";
    try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
            runningMode: "VIDEO", numHands: 1
        });

        await loadInitialData();

        statusElement.textContent = "Ready.";

        webcamButton.disabled = false;


    } catch (error) {
        console.error("Error loading HandLandmarker:", error);
        statusElement.textContent = "Error loading Hand Landmark model!";
        webcamButton.disabled = true;
        classifyButton.disabled = true;
        trainButton.disabled = true;
        saveJsonButton.disabled = true;
        saveButtons.forEach(b => b.disabled = true);
        accuracyStatusElement.textContent = "Accuracy: N/A";
    }
};


async function enableCam() {
    if (!handLandmarker) { console.log("Wait! HandLandmarker not loaded."); return; }
    if (webcamRunning) return;

    webcamRunning = true;
    webcamButton.textContent = "STOP WEBCAM";
    classifyButton.disabled = !isTrained;
    saveButtons.forEach(button => button.disabled = false);

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        if (!video.hasAttribute('data-listener-added')) {
            video.addEventListener("loadeddata", predictWebcam);
            video.setAttribute('data-listener-added', 'true');
        }
    } catch (error) {
        console.error("Error accessing webcam:", error);
        webcamRunning = false; webcamButton.textContent = "START WEBCAM";
        classifyButton.disabled = true;
        saveButtons.forEach(button => button.disabled = true);
    }
}

function stopWebcam() {
    webcamRunning = false;
    webcamButton.textContent = "START WEBCAM";
    classifyButton.disabled = true;
    saveButtons.forEach(button => button.disabled = true);

    const stream = video.srcObject;
    if (stream) { const tracks = stream.getTracks(); tracks.forEach(track => track.stop()); video.srcObject = null; }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    predictionElement.textContent = "Prediction: ?";
    results = undefined; lastVideoTime = -1;
}


async function predictWebcam() {
    if (!webcamRunning) return;


    if (video.videoWidth > 0 && video.videoHeight > 0) {
        if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;

            const videoContainer = document.querySelector(".video-canvas-container");
            if (videoContainer) {
                videoContainer.style.width = video.videoWidth + "px";
                videoContainer.style.height = video.videoHeight + "px";
            }


            console.log(`Canvas resized to: ${video.videoWidth}x${video.videoHeight}`);
        }
    }


    if (handLandmarker && video.readyState >= 2) {
        const startTimeMs = performance.now();
        if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;
            try {
                results = await handLandmarker.detectForVideo(video, startTimeMs);
            } catch (detectionError) {
                console.error("Error during hand detection:", detectionError);
                results = undefined;
            }
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        if (results && results.landmarks && results.landmarks.length > 0) {
            for (const landmarks of results.landmarks) {

                if (drawUtils && HandLandmarker.HAND_CONNECTIONS) {
                    drawUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
                    drawUtils.drawLandmarks(landmarks, { color: "#FF0000", radius: 5 });
                } else {
                    console.error("DrawingUtils or HandLandmarker.HAND_CONNECTIONS not available.");
                }
            }
        }
        canvasCtx.restore();
    }

    window.requestAnimationFrame(predictWebcam);
}


webcamButton.addEventListener("click", () => { if (webcamRunning) stopWebcam(); else enableCam(); });

saveButtons.forEach(button => {
    button.addEventListener("click", () => saveCurrentPose(button.getAttribute("data-label")));
});

trainButton.addEventListener("click", trainModel);

classifyButton.addEventListener("click", classifyCurrentPose);

saveJsonButton.addEventListener("click", saveDataToJsonFile);


if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker();
} else {
    console.error("getUserMedia() is not supported.");
    statusElement.textContent = "Webcam not supported.";
    webcamButton.disabled = true; classifyButton.disabled = true; trainButton.disabled = true;
    saveJsonButton.disabled = true; saveButtons.forEach(b => b.disabled = true);
    accuracyStatusElement.textContent = "Accuracy: N/A";
}