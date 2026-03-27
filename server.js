// server.js - Express backend for handling image uploads and running predictions

const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs');           // pure JS version, no native bindings
const jpeg = require('jpeg-js');                  // for decoding jpeg images
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// serve static files from the frontend folder and model directory
app.use(express.static(path.join(__dirname, '../frontend')));
app.use('/model', express.static(path.join(__dirname, 'model')));

// configure multer to store uploads in a temporary folder called 'uploads'
const upload = multer({ dest: 'uploads/' });

// variables to cache model and class map
let model;
const MODEL_DIR = path.join(__dirname, 'model');
let labels = {}; // index -> class name

// helper to load model once (lazy)
// We create a custom IO handler that reads the JSON and weight files from disk
// directly.  This avoids any network/fetch dependencies and works entirely with
// pure JS tfjs in Node.
async function loadModel() {
  if (!model) {
    // read model.json, with existence check
    const jsonPath = path.join(MODEL_DIR, 'model.json');
    if (!fs.existsSync(jsonPath)) {
      throw new Error(`Model not found at ${jsonPath}. Convert your .h5 to TFJS format and place files in the model directory.`);
    }
    const modelJson = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

    // helper to concat multiple weight files into a single ArrayBuffer
    function concatArrayBuffers(buffers) {
      let total = 0;
      for (const buf of buffers) total += buf.byteLength;
      const tmp = new Uint8Array(total);
      let offset = 0;
      for (const buf of buffers) {
        tmp.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
      }
      return tmp.buffer;
    }

    const weightPaths = modelJson.weightsManifest[0].paths;
    const weightBuffers = weightPaths.map(p => {
      const fullPath = path.join(MODEL_DIR, p);
      if (!fs.existsSync(fullPath)) {
        throw new Error(`Weight file not found: ${fullPath}`);
      }
      return fs.readFileSync(fullPath).buffer;
    });
    const weightData = concatArrayBuffers(weightBuffers);

    const handler = {
      load: async () => {
        return {
          modelTopology: modelJson.modelTopology,
          weightData,
          weightSpecs: modelJson.weightsManifest[0].weights,
        };
      }
    };

    model = await tf.loadLayersModel(handler);
    console.log('Model loaded from disk at', MODEL_DIR);
  }
  return model;
}

// read class_indices.json and invert mapping so we can look up label by index
try {
  const raw = fs.readFileSync(path.join(MODEL_DIR, 'class_indices.json'));
  const classIndices = JSON.parse(raw);
  for (const [label, idx] of Object.entries(classIndices)) {
    labels[idx] = label;
  }
} catch (err) {
  console.warn('Could not read class_indices.json, predictions may not map to names', err);
}

// utility to convert jpeg buffer to a tensor
function bufferToTensor(imageBuffer) {
  const decoded = jpeg.decode(imageBuffer, { useTArray: true });
  // decoded.data is a Uint8Array with size width*height*4 (rgba)
  const arr = decoded.data;
  const numChannels = 3;
  // drop alpha channel
  const values = new Uint8Array((arr.length / 4) * numChannels);
  let j = 0;
  for (let i = 0; i < arr.length; i += 4) {
    values[j++] = arr[i];     // R
    values[j++] = arr[i + 1]; // G
    values[j++] = arr[i + 2]; // B
  }
  return tf.tensor3d(values, [decoded.height, decoded.width, numChannels]);
}

// prediction endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // quick check that model exists before doing any processing
    const jsonPath = path.join(MODEL_DIR, 'model.json');
    if (!fs.existsSync(jsonPath)) {
      return res.status(500).json({ error: 'Model files not found on server' });
    }

    const filepath = req.file.path;

    const imageBuffer = fs.readFileSync(filepath);
    let tensor = bufferToTensor(imageBuffer);

    // resize + normalize
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);
    tensor = tensor.expandDims(0);
    tensor = tensor.toFloat().div(tf.scalar(255));

    const model = await loadModel();
    const prediction = model.predict(tensor);

    const scores = prediction.arraySync()[0];
    const maxIndex = scores.indexOf(Math.max(...scores));
    const predictedLabel = labels[maxIndex] || 'Unknown';
    const confidence = scores[maxIndex] * 100;

    fs.unlinkSync(filepath);
    res.json({ label: predictedLabel, confidence: confidence.toFixed(2) });
  } catch (err) {
    console.error('Error during prediction:', err);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

// start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
