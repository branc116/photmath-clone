import * as tf from "@tensorflow/tfjs"
let video = document.getElementById("video") as HTMLVideoElement;
let info = document.getElementById('info');
let container = document.getElementById('container');
let outString = document.getElementById('outString')
let width = 0;
let height = 0;
let qvga = { width: { ideal: 320 }, height: { ideal: 240 } };
let vga = { width: { ideal: 640 }, height: { ideal: 480 } };
let resolution = window.innerWidth < 640 ? qvga : vga;
let streaming = false;
let stream: MediaStream | null = null;
let vc: VideoCapture | null = null;
declare var cv: OpencvType;
let lastFilter = '';
let src: Mat | null = null;
let dstC1: Mat | null = null;
let dstC3: Mat | null = null;
let dstC4: Mat | null = null;
///@ts-ignore
let mat2: Mat;
type UnPromis<T> = T extends Promise<infer G> ? G : never;
let models: UnPromis<ReturnType<typeof loadModels>> | undefined;
const prefix = info?.getAttribute("data-prefix"); // /photomath
const classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '/', '*', '+', '-'];

const transferMinst = async (x: tf.Tensor) => {
  const buff = tf.buffer([x.shape[0], 16]);
  const src = (await x.buffer());
  for (let j = 0; j < x.shape[0]; j++) {
    for (let i = 0; i < 10; i++) {
      buff.set(src.get(j, i), j, i);
    }
    for (let i = 10; i < 16; i++) { buff.set(0.7, j, i); }
  }
  return buff.toTensor();
}
const transfer2 = async (x: tf.Tensor) => {
  const buff = tf.buffer([x.shape[0], 16]);
  const src = (await x.buffer());
  for (let j = 0; j < x.shape[0]; j++) {
    for (let i = 0; i < 10; i++) {
      buff.set(src.get(j, i + 4), j, i);
      if (i < 2) {
        buff.set(src.get(j, i + 14), j, i + 10)
      }
      if (i < 4) {
        buff.set(src.get(j, i), j, i + 12);
      }
    }
  }
  return buff.toTensor();
}
const loadModels = async () => {
  const modelsP = [{ uri: "models/minstModelJs/model.json", transfer: transferMinst }, { uri: "models/model2Js/model.json", transfer: transfer2 }].map(async i => {
    try {
      let req = await fetch(i.uri);
      let txt = await req.text();
      let config: tf.ModelAndWeightsConfig = JSON.parse(txt)
      config.weightsManifest![0].paths[0] = prefix + config.weightsManifest![0].paths[0];
      let model = await tf.models.modelFromJSON(config);
      return { model, transfer: i.transfer };
    } catch (ex) {
      console.log(ex);
      throw ex;
    }
  });
  return await Promise.all(modelsP)
}

const startCamera = async () => {
  if (streaming) return;
  info!.innerHTML = "TU";
  const devs = navigator.mediaDevices.getSupportedConstraints();
  info!.innerHTML = JSON.stringify(devs);
  const s = await navigator.mediaDevices.getUserMedia({ video: { ...resolution, facingMode: "environment" }, audio: false });

  stream = s;
  video.srcObject = s;
  setInterval(() => {
    if (!s)
      return;
    const track = s.getTracks()[0];
    if (!track)
      return;

    const constraints = track.getConstraints();
    if (!constraints)
      return;
    // info!.innerHTML = JSON.stringify(constraints);
    ///@ts-ignore
    constraints.focusMode = "single-shot";
    track.applyConstraints(constraints).catch(i => {
      info!.innerHTML = JSON.stringify(i)
    });
  }, 3000);
}
const startVideo = () => {
  video.play();
  video.addEventListener("canplay", function (ev) {
    if (!streaming) {
      height = video?.videoHeight ?? 0;
      width = video?.videoWidth ?? 0;
      video.setAttribute("width", width.toString());
      video.setAttribute("height", height.toString());
      streaming = true;
      vc = new cv.VideoCapture(video);
    }
    startVideoProcessing();
  }, false);
}
function startVideoProcessing() {
  if (!streaming) { console.warn("Please startup your webcam"); return; }
  stopVideoProcessing();
  src = new cv.Mat(height, width, cv.CV_8UC4);
  dstC1 = new cv.Mat(height, width, cv.CV_8UC1);
  dstC3 = new cv.Mat(height, width, cv.CV_8UC3);
  dstC4 = new cv.Mat(height, width, cv.CV_8UC4);
  mat2 = new cv.Mat(28, 28, cv.CV_8UC1);
  requestAnimationFrame(processVideo);
}

const shit = () => {
  function passThrough(src: Mat) {
    return src;
  }

  function gray(src: Mat) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    return dstC1;
  }

  function hsv(src: Mat) {
    cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
    cv.cvtColor(dstC3!, dstC3, cv.COLOR_RGB2HSV);
    return dstC3;
  }

  function canny(src: Mat) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    cv.Canny(dstC1!, dstC1, 1, 1, 1, 1);
    return dstC1;
  }

  function inRange(src: Mat) {
    let lowValue = 1;
    let lowScalar = new cv.Scalar(lowValue, lowValue, lowValue, 255);
    let highValue = 1;
    let highScalar = new cv.Scalar(highValue, highValue, highValue, 255);
    let low = new cv.Mat(height, width, src.type(), lowScalar);
    let high = new cv.Mat(height, width, src.type(), highScalar);
    cv.inRange(src, low, high, dstC1);
    low.delete(); high.delete();
    return dstC1;
  }

  function threshold(src: Mat) {
    cv.threshold(src, dstC4, 1, 200, cv.THRESH_BINARY);
    return dstC4;
  }

  function adaptiveThreshold(src: Mat) {
    let mat = new cv.Mat(height, width, cv.CV_8U);
    cv.cvtColor(src, mat, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(mat, dstC1, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, Number(1), 2);
    mat.delete();
    return dstC1;
  }

  function gaussianBlur(src: Mat) {
    cv.GaussianBlur(src, dstC4, { width: 1, height: 1 }, 0, 0, cv.BORDER_DEFAULT);
    return dstC4;
  }

  function bilateralFilter(src: Mat) {
    let mat = new cv.Mat(height, width, cv.CV_8UC3);
    cv.cvtColor(src, mat, cv.COLOR_RGBA2RGB);
    cv.bilateralFilter(mat, dstC3, 1, 1, 1, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC3;
  }

  function medianBlur(src: Mat) {
    cv.medianBlur(src, dstC4, 1);
    return dstC4;
  }

  function sobel(src: Mat) {
    var mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Sobel(mat, dstC1, cv.CV_8U, 1, 0, 1, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
  }

  function scharr(src: Mat) {
    var mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Scharr(mat, dstC1, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
  }

  function laplacian(src: Mat) {
    var mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, 1, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
  }

  const contoursColor: [[number, number, number, number]] = [[1, 2, 3, 4]];
  for (let i = 0; i < 10000; i++) {
    contoursColor.push([Math.round(Math.random() * 255), Math.round(Math.random() * 255), Math.round(Math.random() * 255), 0]);
  }
  function calcHist(src: Mat) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    let srcVec = new cv.MatVector();
    srcVec.push_back(dstC1!);
    let scale = 2;
    let channels = [0];
    const histSize = [src.cols / scale];
    const ranges = [0, 255];
    let hist = new cv.Mat();
    const mask = new cv.Mat();
    const color = new cv.Scalar(0xfb, 0xca, 0x04, 0xff);
    cv.calcHist(srcVec, channels, mask, hist, histSize, ranges);
    let result = cv.minMaxLoc(hist, mask);
    var max = result.maxVal;
    cv.cvtColor(dstC1!, dstC4, cv.COLOR_GRAY2RGBA);
    // draw histogram on src
    for (var i = 0; i < histSize[0]; i++) {
      var binVal = hist.data32F[i] * src.rows / max;
      cv.rectangle(dstC4!, { x: i * scale, y: src.rows - 1 }, { x: (i + 1) * scale - 1, y: src.rows - binVal / 3 }, color, cv.FILLED);
    }
    srcVec.delete();
    mask.delete();
    hist.delete();
    return dstC4;
  }

  function equalizeHist(src: Mat) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY, 0);
    cv.equalizeHist(dstC1!, dstC1);
    return dstC1;
  }

  let base: Mat;

  function backprojection(src: Mat) {
    if (lastFilter !== 'backprojection') {
      if (base instanceof cv.Mat)
        base.delete();
      base = src.clone();
      cv.cvtColor(base, base, cv.COLOR_RGB2HSV, 0);
    }
    cv.cvtColor(src, dstC3, cv.COLOR_RGB2HSV, 0);
    let baseVec = new cv.MatVector(), targetVec = new cv.MatVector();
    baseVec.push_back(base); targetVec.push_back(dstC3!);
    let mask = new cv.Mat(), hist = new cv.Mat();
    let channels = [0], histSize = [50];
    let ranges = [1, 1];
    cv.calcHist(baseVec, channels, mask, hist, histSize, ranges);
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX);
    cv.calcBackProject(targetVec, channels, hist, dstC1, ranges, 1);
    baseVec.delete();
    targetVec.delete();
    mask.delete();
    hist.delete();
    return dstC1;
  }

  function erosion(src: Mat) {
    let kernelSize = 1;
    let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
    let color = new cv.Scalar();
    cv.erode(src, dstC4, kernel, { x: -1, y: -1 }, 1, cv.BORDER_TRANSPARENT, color);
    kernel.delete();
    return dstC4;
  }

  function dilation(src: Mat) {
    let kernelSize = 1;
    let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
    let color = new cv.Scalar();
    cv.dilate(src, dstC4, kernel, { x: -1, y: -1 }, 1, cv.BORDER_CONSTANT, color);
    kernel.delete();
    return dstC4;
  }

  function morphology(src: Mat) {
    let kernelSize = 1;
    let kernel = cv.getStructuringElement(1, { width: kernelSize, height: kernelSize });
    let color = new cv.Scalar();
    let op = Number(1);
    let image = src;
    if (op === cv.MORPH_GRADIENT || op === cv.MORPH_TOPHAT || op === cv.MORPH_BLACKHAT) {
      cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
      image = dstC3!;
    }
    cv.morphologyEx(image, dstC4, op, kernel, { x: -1, y: -1 }, 1, 1, color);
    kernel.delete();
    return dstC4;
  }
}
function contours(src: Mat) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  cv.threshold(dstC1!, dstC4, 120, 200, cv.THRESH_BINARY);
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(dstC4!, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE, { x: 0, y: 0 });
  dstC3?.delete();
  dstC3 = cv.Mat.ones(height, width, cv.CV_8UC3);
  for (let i = 0; i < contours.size(); ++i) {
    const color = [i, i, i, i];
    cv.drawContours(dstC3, contours, i, color, 1, cv.LINE_8, hierarchy);
  }
  contours.delete(); hierarchy.delete();
  return dstC3;
}
function* getBoundinigBoxes(src: Mat, cb: (out: Mat) => void) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  cv.threshold(dstC1!, dstC4, 120, 200, cv.THRESH_BINARY);
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(dstC4!, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE, { x: 0, y: 0 });
  dstC3?.delete();
  dstC3 = cv.Mat.ones(28, 28, cv.CV_8UC3);
  let boxes: ReturnType<typeof cv.boundingRect>[] = [];
  for (let i = 0; i < contours.size(); ++i) {
    boxes.push(cv.boundingRect(contours.get(i)));
  }
  boxes = boxes.filter(i => Math.max(i.width, i.height) > 14).sort(i => i.x);
  for (let i = 0; i < boxes.length; ++i) {
    yield { box: boxes[i], mat: dstC4!.rowRange(boxes[i].y, boxes[i].y + boxes[i].height).colRange(boxes[i].x, boxes[i].x + boxes[i].width) };

  }
  for (let i = 0; i < boxes.length; ++i) {
    var a = new cv.Scalar(255);
    cv.rectangle(dstC1!, { x: boxes[i].x, y: boxes[i].y }, { x: boxes[i].x + boxes[i].width, y: boxes[i].y + boxes[i].height }, a);
  }
  contours.delete(); hierarchy.delete();
  cb(dstC1!);
}
async function* inferImages(src: ReturnType<typeof getBoundinigBoxes>) {
  const arr: number[] = [];
  let k = 0;
  for (const i of src) {
    cv.resize(i.mat, mat2, { width: 28, height: 28 }, cv.INTER_LINEAR);
    arr.push(...mat2.data8S);
    k++;
  }
  const v = tf.tensor4d(arr, [k, 28, 28, 1]).mul(-1).add(255).mul(1 / 255);
  let res = tf.ones([k, 16]);
  for (let j = 0; j < models!.length; j++) {
    let pred = models![j].model.predict(v);
    if (!("length" in pred)) {
      const p = pred.exp();
      p.div(p.norm("euclidean", 0))
      res = res.mul(await models![j].transfer(p.div(p.norm())));

    }
  }
  const d = await res.argMax(0).data();
  for (let i = 0; i < k; i++) {
    yield classes[d[i]];
  }
}
const processVideo = async () => {
  if (!src) return;
  try {
    vc!.read(src);
    const bb = getBoundinigBoxes(src, (bb) => cv.imshow("canvasOutput", bb));
    let s = "";
    for await (const i of inferImages(bb)) {
      s += i;
    }
    try {
      s += ` = ${eval(s)}`;
    } catch (ex) {
      s += " = :(";
    }
    outString!.innerText = s;
  } catch (ex) {
    console.log(ex);
    info!.innerText = ex.toString();
  }
  requestAnimationFrame(processVideo);
}

function stopVideoProcessing() {
  if (src != null && !src.isDeleted()) src.delete();
  if (dstC1 != null && !dstC1.isDeleted()) dstC1.delete();
  if (dstC3 != null && !dstC3.isDeleted()) dstC3.delete();
  if (dstC4 != null && !dstC4.isDeleted()) dstC4.delete();
}

function stopCamera() {
  if (!streaming) return;
  stopVideoProcessing();
  (document.getElementById("canvasOutput") as HTMLCanvasElement).getContext("2d")!.clearRect(0, 0, width, height);
  video.pause();
  video.srcObject = null;
  stream!.getVideoTracks()[0].stop();
  streaming = false;
}
(window as any)["opencvIsReady"] = async (num: number) => {
  if (num === 0)
    return;
  try {
    console.log('OpenCV.js is ready');
    ///@ts-ignore
    if (!featuresReady) {
      info!.innerHTML = 'Requred features are not ready.';
      return;
    }
    models = await loadModels();
    await startCamera();
    startVideo();
  } catch (ex) {
    info!.innerText += ex.toString();
  }
}