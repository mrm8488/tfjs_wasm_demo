import "@tensorflow/tfjs-backend-wasm";
import * as tf from "@tensorflow/tfjs";

const IMAGENET_V2_CLASSES = require("./imagenet_v2_classes");

$("input:radio[name=mode]").change(function() {
  let url;
  if (this.value == "no-wasm") {
    url = new URL(location.href);
    url.searchParams.delete("wasm");
    url.searchParams.set("wasm", false);
    location.href = url;
  } else {
    url = new URL(location.href);
    url.searchParams.delete("wasm");
    url.searchParams.set("wasm", true);
    location.href = url;
  }
});

function status(text, elementId = "status") {
  document.getElementById(elementId).textContent = text;
}

$(document).ready(function() {
  main();
});

async function main() {
  const searchParams = new URLSearchParams(location.search);

  if (!searchParams.get("wasm") || searchParams.get("wasm") === "false") {
    document.getElementById("wasm").checked = false;
    document.getElementById("no-wasm").checked = true;
  } else {
    document.getElementById("no-wasm").checked = false;
    document.getElementById("wasm").checked = true;
    await tf.setBackend("wasm");
  }

  status("Loading the model...");
  const model = await tf.loadGraphModel(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2",
    { fromTFHub: true }
  );
  const imgElement = document.getElementById("img");
  const startTime1 = performance.now();
  status("Model loaded!");

  const imgTensor = tf.browser.fromPixels(imgElement);
  const alignCorners = true;
  const newSize = [224, 224];
  const imgResized = tf.image
    .resizeBilinear(imgTensor, newSize, alignCorners)
    .expandDims();

  const startTime2 = performance.now();
  const resultData = await model.predict(imgResized).dataSync();

  let winnerProbability = -Infinity;
  let winnerPArray = [];

  resultData.forEach((probability, index) => {
    if (probability > winnerProbability) {
      winnerPArray.push(index);
      winnerProbability = probability;
    }
  });
  const top5Predictions = winnerPArray.slice(-5);
  console.log(top5Predictions);

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(
    `Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`
  );

  let predictionsText = "";

  for (let i = 4; i >= 0; i--) {
    predictionsText += `${IMAGENET_V2_CLASSES[top5Predictions[i]]},
        \n`;
  }
  status(predictionsText, "predictions");
}
