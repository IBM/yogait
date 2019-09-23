import { cocoColors, cocoParts } from './coco-common.js'
import { motivationalLines } from './motivation-lines.js'
const lineWidth = 2;
const pointRadius = 4;
const timerLength = 10;

var initialized = false;
var coordinates = new Array();
var canvas;
var currentPrediction;
var targetPose;
var currentConfidence = 0;
var yogaSession;
var currentState = 'waiting';
var poseCanvas;
var targetCanvas;
var textCanvas;

let poseNames = ['y','lunge','warrior']

let overlaySize = {
    width: 640,
    height: 480
}

setup();

// Run setup. Attaches a function to a button
async function setup() {
    console.info('setup');

    let button = document.getElementById("webcamButton");
    button.addEventListener("click", start)

}

/**
 *  Loads the face detector model and creates canvas to display webcam and model results.
 */
function start() {
    console.info('button clicked to start');

    if (initialized) {
        console.log('initialized');
        return;
    }

    // this canvas is where we send the video stream to
    canvas = document.getElementById("canvas");
    canvas.classList.toggle("hide");

    // this canvas is where we draw lines and points
    poseCanvas = document.getElementById("pose-canvas");
    poseCanvas.classList.toggle("hide");

    // this canvas is where we show the target pose
    targetCanvas = document.getElementById("target-canvas");
    targetCanvas.classList.toggle("hide");

    // this canvas is where we draw the timer and instructions
    textCanvas = document.getElementById("text-canvas");
    textCanvas.classList.toggle("hide");

    let button = document.getElementById("webcamButton");
    button.classList.add("hide");

    window.ctx = canvas.getContext('2d', {
        alpha: false
    });


    // set the target pose to be the first in the array
    targetPose = poseNames[0];

    // this lets us do state transitions
    yogaSession = new yogaWrapper(textCanvas);
    yogaSession.setTarget(targetPose);
    yogaSession.generatePrompt();

    var mycamvas = new camvas(window.ctx, processFrame, logTimer, yogaSession);

    // this function gets predictions and draws the lines over the video as well as changes the prediction text
    var predicter = new predicting(canvas, sendImage, poseCanvas, textCanvas, yogaSession);
    
    initialized = true;
}

/**
 * Used by the camvas object to draw the timer or prompt every time the video frame updates (rather than every time there's a prediction)
 * @param {yogaWrapper} wrapper   Object that we pull information from.
 */
function logTimer(wrapper) {
    if (wrapper.enableTiming) {
        // log the timer
        var value =  Date.now() - wrapper.startTime;
        wrapper.ctx.clearRect(0, 0, wrapper.textCanvas.width, wrapper.textCanvas.height);
        wrapper.ctx.fillText(Number((value / 1000).toFixed(1)) + ' seconds', wrapper.textCanvas.width / 2, wrapper.textCanvas.height / 2);
    } else {
        // if we can't log the timer, display a prompt
        wrapper.ctx.clearRect(0, 0, wrapper.textCanvas.width, wrapper.textCanvas.height);
        // pick a random prompt
        wrapper.ctx.fillText(wrapper.prompt, wrapper.textCanvas.width / 2, wrapper.textCanvas.height / 2);
    }
}

/**
 * Contains the state transition logic and timing information.
 * @param {Canvas} textCanvas   Canvas that we send prompt and timer information to.
 */
class yogaWrapper {

    constructor(textCanvas) {
        this.textCanvas = textCanvas;
        this.targetValue = null;
        this.startTime = 0;
        this.enableTiming = false;
        this.ctx = textCanvas.getContext('2d');
        this.ctx.font = "26px IBM Plex Sans";
        this.ctx.textAlign = "center";
        this.prompt = null;
    }

    generatePrompt() {
        var rand = Math.floor(Math.random() * (motivationalLines.length));
        String.prototype.format = function() {
            var a = this;
            for (var k in arguments) {
              a = a.replace("{" + k + "}", arguments[k])
            }
            return a
          }
        this.prompt = motivationalLines[rand].format(this.targetValue);
    }

    clickTimer() {
        this.startTime = Date.now();
        this.enableTiming = !this.enableTiming;
    }

    setTarget(s) {
        this.targetValue = s;
    }

    getStartTime() {
        return this.startTime;
    }
    getTarget() {
        return this.targetValue;
    }
}


/**
 * Runs the predictions and state transition logic.
 * @param {Canvas} canvas           Canvas on which to send the video.
 * @param {Function} sendImage      Function that sends the image to the server.
 * @param {Canvas} poseCanvas       Canvas on which to draw the poses.
 * @param {Canvas} textCanvas       Canvas on which to draw the text.
 * @param {yogaWrapper} yogaSession Object that contains all of the state information.
 */
function predicting(canvas, sendImage, poseCanvas, textCanvas, yogaSession) {
    var self = this;
    this.send = sendImage;
    this.canvas = canvas;
    this.poseCanvas = poseCanvas;
    this.textCanavs = textCanvas;
    this.yogaSession = yogaSession;

    // send image to server, wait until it returns a prediction
    var self = this;
    var last = Date.now();
    var loop = async function () {
        var dt = Date.now() - last;
        let arr = await self.send(canvas);
        if (typeof(arr) !== 'undefined') {
            var pose = arr[0];
            currentPrediction = arr[1];
            currentConfidence = arr[2];

            // draw all poses
            poseCanvas.getContext('2d').clearRect(0, 0, overlaySize.width, overlaySize.height)
            var i;
            for (i = 0; i < pose['predictions'].length; i++) {
                drawBodyParts(poseCanvas.getContext('2d'), pose['predictions'][i]['body_parts'], cocoParts, cocoColors)
                drawPoseLines(poseCanvas.getContext('2d'), pose['predictions'][i]['pose_lines'], cocoColors)
            }

            // logic for state transitions
            if (currentState === 'waiting') {
                if (currentPrediction === yogaSession.targetValue && currentConfidence > 90) {
                    console.log('\tthe posing has begun!');
                    yogaSession.clickTimer();
                    currentState = 'posing';
                } else {
                    console.log('waiting for: ' + targetPose);
                }
            }
            else {
                var poseTime = Date.now() - yogaSession.getStartTime();
                if (currentPrediction === yogaSession.getTarget() && currentConfidence > 90) {
                    if (poseTime > timerLength * 1000) {
                        console.log('\t\ttransition to next pose!');
                        currentState = 'waiting';
                        targetPose = poseNames[poseNames.indexOf(targetPose)+1];
                        console.log('new pose: ' + targetPose);
                        yogaSession.setTarget(targetPose);
                        yogaSession.clickTimer();
                        yogaSession.generatePrompt();
                        // display congratulatory message
                        setTimeout(function() {
                            this.textCanvas.getContext('2d').clearRect(0, 0, wrapper.textCanvas.width, wrapper.textCanvas.height);
                            this.textCanvas.getContext('2d').fillText('Great job :)', wrapper.textCanvas.width / 2, wrapper.textCanvas.height / 2);
                        }, 2000);
                    } else {
                        console.log('\tholding pose');
                    }
                }
                if (currentConfidence < 90) {
                    currentState = 'waiting';
                    yogaSession.clickTimer();
                }
            }
        } else {
            console.log('CANT DRAW YET');
        }
        last = Date.now();
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

}

/**
 * Draws the specified pose on the pose canvas.
 * @param {Canvas} canvas       Canvas on which to draw the pose.
 * @param {String} pose_name    Name of the file to be drawn.
 */

async function displayPose(canvas, pose_name) {
    const start = (new Date()).getTime();
    drawing = new Image();
    drawing.src = "images/"+pose_name+".png";
    drawing.onload = function() {
        canvas.getContext('2d').drawImage(drawing,
            canvas.width / 2 - drawing.width * 0.8 / 2, canvas.height / 2 - drawing.height * 0.8 / 2,
            drawing.width * 0.8, drawing.height * 0.8);
    };
}


/**
 * Runs every frame update. Grab the image from the webcam, run face detection, then crop
 * images for faces and send those images to the model.
 * @param {Object} video    Video object.
 * @param {Number} dt       Time elapsed between frames.
 */

async function processFrame(video, dt) {
    const start = (new Date()).getTime();
    // render the video frame to the canvas element and extract RGBA pixel data
    window.ctx.drawImage(video, 0, 0);
}


/**
 *  Sends an image to the MAX server and receive a prediction in response
 */
function sendImage(canvas) {
    // get the image from the canvas
    var endpoint = 'http://localhost:5000/model/predict';

    return new Promise(function (resolve, reject) {
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob);
            formData.append('type', 'image/jpeg');

            const res = await fetch(endpoint, {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'application/json'
                }
            });

            var posePred = await res.json();

            if (typeof posePred['predictions'][0] === 'undefined') {
                reject('posePred undefined!')
            } else {
                // send coordinates to the svm service
                coordinates = posePred['predictions'][0]['body_parts']
                const formData = new FormData();
                formData.append('file', JSON.stringify(coordinates));
                formData.append('type', 'application/json');

                const res = await fetch('/svm', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                var result = await res.text();

                var values = result.split(',');
                var pose = values[0];
                var confidence = values[1];
                resolve([posePred, pose, parseFloat(confidence)])
            }
        }, 'image/jpeg', 1.0);
    }).catch(function (error) {
        console.log(error);
    });
}


/**
 * draw point on given canvas
 *
 * @param {CanvasRenderingContext2D} canvasCtx - the canvas rendering context to draw point
 * @param {Integer} x - the horizontal value of point
 * @param {Integer} y - the vertical value of point
 * @param {String} c - the color value for point
 */
function drawPoint(canvasCtx, x, y, c = 'black', sx = 1, sy = 1) {
    canvasCtx.beginPath()
    canvasCtx.arc(x * sx, y * sy, pointRadius, 0, 2 * Math.PI)
    canvasCtx.fillStyle = c
    canvasCtx.fill()
}

/**
 * Draws a line on a canvas
 *
 * @param {CanvasRenderingContext2D} canvasCtx - the canvas rendering context to draw point
 * @param {Integer} x1 - the horizontal value of first point
 * @param {Integer} y1 - the vertical value of first point
 * @param {Integer} x2 - the horizontal value of first point
 * @param {Integer} y2 - the vertical value of first point
 * @param {String} c - the color value for line
 */
function drawLine(canvasCtx, x1, y1, x2, y2, c = 'black', sx = 1, sy = 1) {
    canvasCtx.beginPath()
    canvasCtx.moveTo(x1 * sx, y1 * sy)
    canvasCtx.lineTo(x2 * sx, y2 * sy)
    canvasCtx.lineWidth = lineWidth
    canvasCtx.strokeStyle = c
    canvasCtx.stroke()
}

/**
 * Draws the pose lines (i.e., skeleton)
 *
 * @param {CanvasRenderingContext2D} ctx - the canvas rendering context to draw pose lines
 * @param {Array} poseLines - array of coordinates corresponding to the pose lines
 * @param {Array} colors - array of RGB values of colors to use for drawing pose lines
 */
function drawPoseLines(ctx, poseLines, colors, scale = [1, 1]) {
    poseLines.forEach((l, j) => {
        var data = JSON.stringify(Object.values(l));
        data = data.replace('[[','');
        data = data.replace(']]','');
        var lines = data.split(',');
        let color = `rgb(${colors[j].join()})`
        drawLine(ctx, lines[0], lines[1], lines[2], lines[3], color, scale[0], scale[1])
    })
}

/**
 * Draws the left and right wrists keypoints
 *
 * @param {CanvasRenderingContext2D} ctx - the canvas rendering context to draw pose lines
 * @param {Array} bodyParts - array of objects containing body part info
 * @param {Array} partsToDraw - array of the body parts to draw
 * @param {Array} colors - array of RGB values of colors to use for drawing pose lines
 */
function drawBodyParts(ctx, bodyParts, partsToDraw, colors, scale = [1, 1]) {
    bodyParts.forEach(p => {
        if (!partsToDraw || partsToDraw.includes(p['part_name'])) {
            let color = `rgb(${colors[p['part_id']]})`
            drawPoint(ctx, p['x'], p['y'], color, scale[0], scale[1])
        }
    })
}