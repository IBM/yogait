//
// Copyright 2018 IBM Corp. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const modelUrl = 'public/model/tensorflowjs_model.pb'
const weightsUrl = 'public/model/weights_manifest.json'
var theModel

// /**
//  * Loads the TFJS model.
//  */
// async function loadModel() {
//     tf.disableDeprecationWarnings();
//     theModel = await tf.loadFrozenModel(modelUrl, weightsUrl);
// }

/**
 * Runs the model on a single frame.
 * @param {Object} video                    Video object.
 * @param {Array} detectResult              Array containing results from face detection model.
 * @return {Int32Array}                     Returns the results from the model in an array.
 */
async function runPrediction(video, detectResult) {
    let ppl_count = 0;
    var img = tf.zeros([1, 64, 64, 3]);
    var age_rst = new Array(detectResult.length)
    var x = new Array(detectResult.length)
    var y = new Array(detectResult.length)
    var bbxwidth = new Array(detectResult.length)
    var bbxheight = new Array(detectResult.length)
    for (let i = 0; i < detectResult.length; i++) {
        ppl_count += 1
        x[i] = detectResult[i][0];
        y[i] = detectResult[i][1];
        bbxwidth[i] = detectResult[i][2];
        bbxheight[i] = detectResult[i][3];
        // crop detected faces
        tmp = await cropTensor(video, x[i], y[i], bbxwidth[i], bbxheight[i]);
        if (i == 0) {
            img = tmp
        }
        else {
            img = tf.concat([img, tmp])
        }
    }
    // Inference
    let output = tf.tidy(() => theModel.predict(img))
    age_rst = output.toInt()
    return age_rst
}

/**
 * Creates a tensor to be used as input from the original images and coordinates that specify
 * where a face is located.
 * @param {*} imageInput 
 * @param {Number} x             Face bounding box x coordinate.
 * @param {Number} y             Face bounding box y coordinate.
 * @param {Number} w             Face bounding box width.
 * @param {Number} h             Face bounding box height.
 * 
 * @return {tf.Tensor}           A 2-D (64,64) shaped tensor containing a cropped face.
 */async function cropTensor(imageInput, x, y, w, h) {
    return tf.tidy(() => {

        // read input image into tensor
        let inputTensor = tf.browser.fromPixels(imageInput)

        // crop face tensor
        inputTensor = inputTensor.slice([y, x], [h, w])
        inputTensor = tf.image.resizeBilinear(inputTensor, ([64, 64]))
        inputTensor = inputTensor.toFloat();
        let cropped_face_tensor = tf.expandDims(inputTensor);
        return cropped_face_tensor
    })
}
