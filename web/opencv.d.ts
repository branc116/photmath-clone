type VideoCapture = {
    read: (dst: Mat) => void;
}
interface Mat {
    
    new(height?: number, width?: number, type?: number, scalar?: Scalar): Mat;
    rowRange: (start: number, count: number) => Mat
    colRange: (start: number, count: number) => Mat
    ones: (height: number, width: number, type: number) => Mat;
    delete: () => void;
    isDeleted: () => boolean;
    data32F: Array<number>;
    data8S: number[];
    clone: () => Mat;
    type: () => number;
    cols: number;
    rows: number;
}
interface Scalar {
    new(v0?: any,v1?: any,v2?: any,v3?: any): Scalar;
    new(): Scalar;
    delete: () => void;
}
type IntVector = {
}
type FloatVector = {
}
type DoubleVector = {
}
type PointVector = {
}
type MatVector = {
    size: () => number;
    delete: () => number;
    get: (i: number) => Mat
    push_back: (mat: Mat) => void;
}
type RectVector = {
}


type FS_createFolder = {
}
type FS_createPath = {
}
type FS_createDataFile = {
}
type FS_createPreloadedFile = {
}
type FS_createLazyFile = {
}
type FS_createLink = {
}
type FS_createDevice = {
}
type FS_unlink = {
}
type Canny = {
}
type Canny1 = {
}
type GaussianBlur = {
}
type HoughCircles = {
}
type HoughLines = {
}
type HoughLinesP = {
}
type Laplacian = {
}
type Scharr = {
}
type Sobel = {
}
type OpencvType = {
    VideoCapture: new(element: HTMLVideoElement) => VideoCapture;
    Mat: Mat
    wasmBinaryFile: string;
    _main: () => any;
    read: (url: any) => any;
    readAsync: (url: any,onload: any,onerror: any) => any;
    arguments: object;
    print: (x: any) => any;
    printErr: (x: any) => any;
    setWindowTitle: (title: any) => any;
    load: (f: any) => any;
    thisProgram: string;
    quit: (status: any,toThrow: any) => any;
    preRun: object;
    postRun: object;
    Runtime: object;
    ccall: (ident: any,returnType: any,argTypes: any,args: any,opts: any) => any;
    cwrap: (ident: any,returnType: any,argTypes: any) => any;
    setValue: (ptr: any,value: any,type: any,noSafe: any) => any;
    getValue: (ptr: any,type: any,noSafe: any) => any;
    ALLOC_NORMAL: number;
    ALLOC_STACK: number;
    ALLOC_STATIC: number;
    ALLOC_DYNAMIC: number;
    ALLOC_NONE: number;
    allocate: (slab: any,types: any,allocator: any,ptr: any) => any;
    getMemory: (size: any) => any;
    Pointer_stringify: (ptr: any,length: any) => any;
    AsciiToString: (ptr: any) => any;
    stringToAscii: (str: any,outPtr: any) => any;
    UTF8ArrayToString: (u8Array: any,idx: any) => any;
    UTF8ToString: (ptr: any) => any;
    stringToUTF8Array: (str: any,outU8Array: any,outIdx: any,maxBytesToWrite: any) => any;
    stringToUTF8: (str: any,outPtr: any,maxBytesToWrite: any) => any;
    lengthBytesUTF8: (str: any) => any;
    stackTrace: () => any;
    reallocBuffer: (size: any) => any;
    wasmMemory: object;
    HEAP8: object;
    HEAP16: object;
    HEAP32: object;
    HEAPU8: object;
    HEAPU16: object;
    HEAPU32: object;
    HEAPF32: object;
    HEAPF64: object;
    HEAP: undefined;
    buffer: object;
    addOnPreRun: (cb: any) => any;
    addOnInit: (cb: any) => any;
    addOnPreMain: (cb: any) => any;
    addOnExit: (cb: any) => any;
    addOnPostRun: (cb: any) => any;
    intArrayFromString: (stringy: any,dontAddNull: any,length: any) => any;
    intArrayToString: (array: any) => any;
    writeStringToMemory: (string: any,buffer: any,dontAddNull: any) => any;
    writeArrayToMemory: (array: any,buffer: any) => any;
    writeAsciiToMemory: (str: any,buffer: any,dontAddNull: any) => any;
    addRunDependency: (id: any) => any;
    removeRunDependency: (id: any) => any;
    preloadedImages: object;
    preloadedAudios: object;
    wasmJSMethod: string;
    asmPreload: undefined;
    asm: object;
    STATIC_BASE: number;
    STATIC_BUMP: number;
    _pthread_mutex_unlock: () => any;
    _pthread_mutex_lock: () => any;
    _memset: () => any;
    _pthread_cond_broadcast: () => any;
    _memcpy: () => any;
    _sbrk: () => any;
    _memmove: () => any;
    _llvm_bswap_i32: () => any;
    count_emval_handles: () => any;
    get_first_emval: () => any;
    BindingError: () => any;
    InternalError: () => any;
    FS_createFolder: new(parent: any,name: any,canRead: any,canWrite: any) => FS_createFolder
    FS_createPath: new(parent: any,path: any,canRead: any,canWrite: any) => FS_createPath
    FS_createDataFile: new(parent: any,name: any,data: any,canRead: any,canWrite: any,canOwn: any) => FS_createDataFile
    FS_createPreloadedFile: new(parent: any,name: any,url: any,canRead: any,canWrite: any,onload: any,onerror: any,dontCreateFile: any,canOwn: any,preFinish: any) => FS_createPreloadedFile
    FS_createLazyFile: new(parent: any,name: any,url: any,canRead: any,canWrite: any) => FS_createLazyFile
    FS_createLink: new(parent: any,name: any,target: any,canRead: any,canWrite: any) => FS_createLink
    FS_createDevice: new(parent: any,name: any,input: any,output: any) => FS_createDevice
    FS_unlink: new(path: any) => FS_unlink
    requestFullScreen: (lockPointer: any,resizeCanvas: any,vrDevice: any) => any;
    requestFullscreen: (lockPointer: any,resizeCanvas: any,vrDevice: any) => any;
    requestAnimationFrame: (func: any) => any;
    setCanvasSize: (width: any,height: any,noUpdates: any) => any;
    pauseMainLoop: () => any;
    resumeMainLoop: () => any;
    getUserMedia: () => any;
    createContext: (canvas: any,useWebGL: any,setInModule: any,webGLContextAttributes: any) => any;
    getInheritedInstanceCount: () => any;
    getLiveInheritedInstances: () => any;
    flushPendingDeletes: () => any;
    setDelayFunction: (fn: any) => any;
    UnboundTypeError: () => any;
    wasmTableSize: number;
    wasmMaxTableSize: number;
    asmGlobalArg: object;
    asmLibraryArg: object;
    wasmTable: object;
    __GLOBAL__sub_I_system_cpp: () => any;
    __GLOBAL__sub_I_umatrix_cpp: () => any;
    stackSave: () => any;
    getTempRet0: () => any;
    setThrew: () => any;
    __GLOBAL__sub_I_persistence_cpp: () => any;
    _fflush: () => any;
    ___cxa_is_pointer_type: () => any;
    __GLOBAL__sub_I_trace_cpp: () => any;
    __GLOBAL__sub_I_haar_cpp: () => any;
    ___cxa_demangle: () => any;
    __GLOBAL__sub_I_imgwarp_cpp: () => any;
    stackAlloc: () => any;
    __GLOBAL__sub_I_color_cpp: () => any;
    __GLOBAL__sub_I_bind_cpp: () => any;
    setTempRet0: () => any;
    __GLOBAL__I_000101: () => any;
    _emscripten_get_global_libc: () => any;
    ___getTypeName: () => any;
    __GLOBAL__sub_I_iostream_cpp: () => any;
    ___errno_location: () => any;
    ___cxa_can_catch: () => any;
    _free: () => any;
    runPostSets: () => any;
    __GLOBAL__sub_I_hog_cpp: () => any;
    establishStackSpace: () => any;
    __GLOBAL__sub_I_bindings_cpp: () => any;
    stackRestore: () => any;
    _malloc: () => any;
    __GLOBAL__sub_I_histogram_cpp: () => any;
    _emscripten_replace_memory: () => any;
    dynCall_iiiiiid: () => any;
    dynCall_viiiiddd: () => any;
    dynCall_viiiidiii: () => any;
    dynCall_viiiiiddi: () => any;
    dynCall_viiidiii: () => any;
    dynCall_viiiidiid: () => any;
    dynCall_iiiiiii: () => any;
    dynCall_viiiidddiiii: () => any;
    dynCall_viiiiddi: () => any;
    dynCall_viiiddddi: () => any;
    dynCall_viiidddi: () => any;
    dynCall_viiiiiiiiiii: () => any;
    dynCall_viiiiiiiiiid: () => any;
    dynCall_viiidddd: () => any;
    dynCall_iidi: () => any;
    dynCall_viiddidd: () => any;
    dynCall_vidi: () => any;
    dynCall_viiddii: () => any;
    dynCall_viiddid: () => any;
    dynCall_viiiiidi: () => any;
    dynCall_viiddidddd: () => any;
    dynCall_viiiiiiidd: () => any;
    dynCall_viiiiddiiid: () => any;
    dynCall_viiiiiiidi: () => any;
    dynCall_fii: () => any;
    dynCall_viiidii: () => any;
    dynCall_viiiiidd: () => any;
    dynCall_di: () => any;
    dynCall_viiiiiidiiii: () => any;
    dynCall_viiiidiiddi: () => any;
    dynCall_viiiiddiii: () => any;
    dynCall_vdii: () => any;
    dynCall_diiiiiii: () => any;
    dynCall_dii: () => any;
    dynCall_viiiddiiid: () => any;
    dynCall_viiiidiidd: () => any;
    dynCall_viiiiiiiiiiddi: () => any;
    dynCall_iiiii: () => any;
    dynCall_viiiiidiiiii: () => any;
    dynCall_viiiiidiidd: () => any;
    dynCall_iiiid: () => any;
    dynCall_iiiif: () => any;
    dynCall_iiiiiiii: () => any;
    dynCall_viiddiii: () => any;
    dynCall_iiiiiiiididiii: () => any;
    dynCall_viiidddiii: () => any;
    dynCall_viidiiid: () => any;
    dynCall_viiiiidiiii: () => any;
    dynCall_diiiddi: () => any;
    dynCall_viiididii: () => any;
    dynCall_diiiiiiii: () => any;
    dynCall_viiidiiid: () => any;
    dynCall_viiiddddii: () => any;
    dynCall_viiiiid: () => any;
    dynCall_viiiiddddii: () => any;
    dynCall_viiiiii: () => any;
    dynCall_viiidiiii: () => any;
    dynCall_viiiiiidi: () => any;
    dynCall_viiiiiidiii: () => any;
    dynCall_fiii: () => any;
    dynCall_viiiidddii: () => any;
    dynCall_viiidd: () => any;
    dynCall_viiidi: () => any;
    dynCall_viiddiddd: () => any;
    dynCall_viiiiiiiiii: () => any;
    dynCall_diiddi: () => any;
    dynCall_diii: () => any;
    dynCall_viiiddd: () => any;
    dynCall_viiiddidddd: () => any;
    dynCall_viiiiiiiiiiid: () => any;
    dynCall_viiiddidd: () => any;
    dynCall_viiidiiiidi: () => any;
    dynCall_viiiddi: () => any;
    dynCall_fiiii: () => any;
    dynCall_iiiiii: () => any;
    dynCall_viiid: () => any;
    dynCall_iiiiij: () => any;
    dynCall_iiiiid: () => any;
    dynCall_viiiidddi: () => any;
    dynCall_viiii: () => any;
    dynCall_viiiii: () => any;
    dynCall_viiiidiiii: () => any;
    dynCall_vid: () => any;
    dynCall_iiidi: () => any;
    dynCall_iiidd: () => any;
    dynCall_vii: () => any;
    dynCall_viiiid: () => any;
    dynCall_viiiiddddi: () => any;
    dynCall_viidd: () => any;
    dynCall_viidi: () => any;
    dynCall_viiidiiddi: () => any;
    dynCall_diiid: () => any;
    dynCall_viiidddii: () => any;
    dynCall_viiiiiiii: () => any;
    dynCall_viiidddiiii: () => any;
    dynCall_viiiiiiid: () => any;
    dynCall_diiii: () => any;
    dynCall_viiiiidiiddi: () => any;
    dynCall_viiiiidii: () => any;
    dynCall_viiiddiddd: () => any;
    dynCall_fiiiii: () => any;
    dynCall_viiiddid: () => any;
    dynCall_viiiiiii: () => any;
    dynCall_viididdi: () => any;
    dynCall_viiiiiid: () => any;
    dynCall_viiiiiiiii: () => any;
    dynCall_iii: () => any;
    dynCall_viiiddii: () => any;
    dynCall_viiididi: () => any;
    dynCall_vdiii: () => any;
    dynCall_viiiiiidii: () => any;
    dynCall_viiiidddiii: () => any;
    dynCall_viii: () => any;
    dynCall_v: () => any;
    dynCall_viid: () => any;
    dynCall_viif: () => any;
    dynCall_vi: () => any;
    dynCall_viiiidiiiidi: () => any;
    dynCall_ii: () => any;
    dynCall_viijii: () => any;
    dynCall_viiiiiiiddi: () => any;
    dynCall_vididdi: () => any;
    dynCall_viiiiiidd: () => any;
    dynCall_vidii: () => any;
    dynCall_viiif: () => any;
    dynCall_viiiddiii: () => any;
    dynCall_viiiiiidiiiii: () => any;
    dynCall_iiii: () => any;
    dynCall_viididii: () => any;
    dynCall_viiiidddd: () => any;
    dynCall_viiiiddii: () => any;
    dynCall_iiid: () => any;
    dynCall_viiiidii: () => any;
    dynCall_viiiidi: () => any;
    dynCall_diiiii: () => any;
    dynCall_diiiid: () => any;
    dynCall_iiiiiiiiiiiii: () => any;
    dynCall_viiiiiiddi: () => any;
    dynCall_iid: () => any;
    dynCall_i: () => any;
    dynCall_diiiiii: () => any;
    dynCall_vididdii: () => any;
    dynCall_viiddi: () => any;
    dynCall_viiiiidiii: () => any;
    dynCall_viididi: () => any;
    dynCall_iiiiiiiii: () => any;
    dynCall_viididdii: () => any;
    dynCall_viiiiidiid: () => any;
    dynCall_viiiidd: () => any;
    dynCall_vidiii: () => any;
    then: (func: any) => any;
    callMain: (args: any) => any;
    run: (args: any) => any;
    exit: (status: any,implicit: any) => any;
    abort: (what: any) => any;
    imread: (imageSource: any) => any;
    imshow: (canvasSource: any,mat: any) => any;
    Range: (start: any,end: any) => any;
    Point: (x: any,y: any) => any;
    Size: (width: any,height: any) => any;
    Rect: () => any;
    RotatedRect: () => any;
    Scalar: Scalar;
    MinMaxLoc: () => any;
    Circle: () => any;
    TermCriteria: () => any;
    matFromArray: (rows: any,cols: any,type: any,array: any) => any;
    matFromImageData: (imageData: any) => any;
    usingWasm: boolean;
    calledRun: boolean;
    stdin: undefined;
    stdout: undefined;
    stderr: undefined;
    IntVector: new () => IntVector;
    FloatVector: new () => FloatVector;
    DoubleVector: new () => DoubleVector;
    PointVector: new () => PointVector;
    MatVector: new () => MatVector;
    RectVector: new () => RectVector;
    rotatedRectPoints: (arg0: any) => any;
    rotatedRectBoundingRect: (arg0: any) => any;
    rotatedRectBoundingRect2f: (arg0: any) => any;
    minEnclosingCircle: (arg0: any) => any;
    minMaxLoc: (hist: Mat, mask: Mat) => {maxVal: number};
    morphologyDefaultBorderValue: () => any;
    CV_MAT_DEPTH: (arg0: any) => any;
    CamShift: (arg0: any, arg1: any, arg2: any) => any;
    meanShift: (arg0: any, arg1: any, arg2: any) => any;
    Canny: (src: Mat, dst: Mat | null, cannyThreshold1: number, cannyThreshold2: number, cannyApertureSize: number, cannyL2Gradient: number) => void
    Canny1: new() => Canny1
    GaussianBlur: (src: Mat, dst: Mat | null, size: { width: number, height: number }, arg1: number, arg2: number, type: number) => void
    HoughCircles: new() => HoughCircles
    HoughLines: new() => HoughLines
    HoughLinesP: new() => HoughLinesP
    Laplacian: (src: Mat, dst: Mat | null, size: Number, laplacianSize: number, arg1: number, arg2: number, borderType: number) => void
    Scharr: (src: Mat, dst: Mat | null, size: number, arg1: number, arg2: number, arg3: number, arg4:number, borderType: number) => void
    Sobel: (src: Mat, dst: Mat | null, type: number, arg1: number, arg2: number, sobelSize: number, arg3: number, arg4: number, borderType: number) => void
    absdiff: (arg0: any, arg1: any, arg2: any) => any;
    adaptiveThreshold: (src: Mat, dst: Mat | null, arg2: number, arg3: number, arg4: number, arg5: number, arg6: number) => void;
    add: () => any;
    addWeighted: () => any;
    approxPolyDP: (arg0: any, arg1: any, arg2: any, arg3: any) => any;
    arcLength: (arg0: any, arg1: any) => any;
    bilateralFilter: (src: Mat, dst: Mat | null, bilateralFilterDiameter: number, bilateralFilterSigma1: number, bilateralFilterSigma2: number, type: number) => void;
    bitwise_and: () => any;
    bitwise_not: () => any;
    bitwise_or: () => any;
    bitwise_xor: () => any;
    blur: () => any;
    boundingRect: (src: Mat) => {x: number, y: number, width: number, height: number};
    boxFilter: () => any;
    calcBackProject: (targetVec: MatVector, channels: number[], hist: Mat, dst: Mat | null, ranges: number[], last: number) => any;
    calcHist: (srcVec: MatVector, channels: number[], mask: Mat, hist: Mat, histSize: number[], ranges: number[]) => void;
    calcOpticalFlowFarneback: (arg0: any, arg1: any, arg2: any, arg3: any, arg4: any, arg5: any, arg6: any, arg7: any, arg8: any, arg9: any) => any;
    calcOpticalFlowPyrLK: () => any;
    cartToPolar: () => any;
    circle: () => any;
    compare: (arg0: any, arg1: any, arg2: any, arg3: any) => any;
    compareHist: (arg0: any, arg1: any, arg2: any) => any;
    connectedComponents: () => any;
    connectedComponentsWithStats: () => any;
    contourArea: () => any;
    convertScaleAbs: () => any;
    convexHull: () => any;
    convexityDefects: (arg0: any, arg1: any, arg2: any) => any;
    copyMakeBorder: () => any;
    cornerHarris: () => any;
    cornerMinEigenVal: () => any;
    countNonZero: (arg0: any) => any;
    cvtColor: (src: Mat, dest: Mat | null, type: number, forth?: number) => void;
    demosaicing: () => any;
    determinant: (arg0: any) => any;
    dft: () => any;
    dilate: (src: Mat, dst: Mat | null, kernel: Mat, point: { x: number, y: number }, something: number, dilationBorderType: number, color: Scalar) => void;
    distanceTransform: () => any;
    distanceTransformWithLabels: () => any;
    divide: () => any;
    divide1: () => any;
    drawContours: (drawTo: Mat, contours: MatVector, i: number, color: number[], type: number, line: number, hierarchy: Mat) => void;
    eigen: () => any;
    ellipse: () => any;
    ellipse1: () => any;
    ellipse2Poly: (arg0: any, arg1: any, arg2: any, arg3: any, arg4: any, arg5: any, arg6: any) => any;
    equalizeHist: (src: Mat, dst: Mat | null) => void;
    erode: (src: Mat, dst: Mat | null, kernel: Mat, point: { x: number, y: number }, something: number, erosionBorderType: number, color: Scalar) => void;
    estimateRigidTransform: (arg0: any, arg1: any, arg2: any) => any;
    exp: (arg0: any, arg1: any) => any;
    filter2D: () => any;
    findContours: (src: Mat, contours: MatVector, hierarchy: Mat, returnType: number, aproxMode: number, position: { x: number, y: number }) => void;
    findTransformECC: () => any;
    fitEllipse: (arg0: any) => any;
    fitLine: (arg0: any, arg1: any, arg2: any, arg3: any, arg4: any, arg5: any) => any;
    flip: (arg0: any, arg1: any, arg2: any) => any;
    gemm: () => any;
    getAffineTransform: (arg0: any, arg1: any) => any;
    getOptimalDFTSize: (arg0: any) => any;
    getPerspectiveTransform: (arg0: any, arg1: any) => any;
    getRotationMatrix2D: (arg0: any, arg1: any, arg2: any) => any;
    getStructuringElement: (morphologyShape: number, size: { width: number, height: number }) => Mat;
    goodFeaturesToTrack: () => any;
    grabCut: () => any;
    groupRectangles: () => any;
    hconcat: (arg0: any, arg1: any) => any;
    inRange: (arg0: any, arg1: any, arg2: any, arg3: any) => any;
    initUndistortRectifyMap: (arg0: any, arg1: any, arg2: any, arg3: any, arg4: any, arg5: any, arg6: any, arg7: any) => any;
    integral: () => any;
    integral2: () => any;
    invert: () => any;
    isContourConvex: (arg0: any) => any;
    kmeans: () => any;
    line: () => any;
    log: (arg0: any, arg1: any) => any;
    magnitude: (arg0: any, arg1: any, arg2: any) => any;
    matchShapes: (arg0: any, arg1: any, arg2: any, arg3: any) => any;
    matchTemplate: () => any;
    max: (arg0: any, arg1: any, arg2: any) => any;
    mean: () => any;
    meanStdDev: () => any;
    medianBlur: (src: Mat, dst: Mat | null, blur: number) => void;
    merge: (arg0: any, arg1: any) => any;
    min: (arg0: any, arg1: any, arg2: any) => any;
    minAreaRect: (arg0: any) => any;
    mixChannels: () => any;
    moments: () => any;
    morphologyEx: (src: Mat, dst: Mat | null, op: number, kernel: Mat, position: { x: number, y: number }, something: number, morphologyBorderType: number, color: Scalar) => dst is Mat;
    multiply: () => any;
    norm: () => any;
    norm1: () => any;
    normalize: (src: Mat, dst: Mat, minVal: number, maxVal: number, normType: number) => void;
    perspectiveTransform: (arg0: any, arg1: any, arg2: any) => any;
    pointPolygonTest: (arg0: any, arg1: any, arg2: any) => any;
    polarToCart: () => any;
    pow: (arg0: any, arg1: any, arg2: any) => any;
    putText: () => any;
    pyrDown: () => any;
    pyrUp: () => any;
    randn: (arg0: any, arg1: any, arg2: any) => any;
    randu: (arg0: any, arg1: any, arg2: any) => any;
    rectangle: (dst: Mat, start: { x: number, y: number }, end: { x: number, y: number }, color: Scalar, filledType?: number) => void;
    reduce: () => any;
    remap: () => any;
    repeat: (arg0: any, arg1: any, arg2: any, arg3: any) => any;
    resize: (src: Mat, dst: Mat, size: {width: number, height: number}, type: number) => void;
    sepFilter2D: () => any;
    setIdentity: () => any;
    setRNGSeed: (arg0: any) => any;
    solve: () => any;
    solvePoly: () => any;
    split: (arg0: any, arg1: any) => any;
    sqrt: (arg0: any, arg1: any) => any;
    subtract: () => any;
    threshold: (src: Mat, dst: Mat | null, threshold1: number, threshold2: number, type: number) => void;
    trace: (arg0: any) => any;
    transform: (arg0: any, arg1: any, arg2: any) => any;
    transpose: (arg0: any, arg1: any) => any;
    undistort: () => any;
    vconcat: (arg0: any, arg1: any) => any;
    warpAffine: () => any;
    warpPerspective: () => any;
    watershed: (arg0: any, arg1: any) => any;
    HOGDescriptor: () => any;
    BackgroundSubtractor: () => any;
    BackgroundSubtractorMOG2: () => any;
    CLAHE: () => any;
    Algorithm: () => any;
    CascadeClassifier: () => any;
} & OpenCV_Enums;
type OpenCV_Enums = {
    ALLOC_NORMAL: 0;
ALLOC_STACK: 1;
ALLOC_STATIC: 2;
ALLOC_DYNAMIC: 3;
ALLOC_NONE: 4;
STATIC_BASE: 1024;
STATIC_BUMP: 2883616;
wasmTableSize: 4274;
wasmMaxTableSize: 4274;
CV_8UC1: 0;
CV_8UC2: 8;
CV_8UC3: 16;
CV_8UC4: 24;
CV_8SC1: 1;
CV_8SC2: 9;
CV_8SC3: 17;
CV_8SC4: 25;
CV_16UC1: 2;
CV_16UC2: 10;
CV_16UC3: 18;
CV_16UC4: 26;
CV_16SC1: 3;
CV_16SC2: 11;
CV_16SC3: 19;
CV_16SC4: 27;
CV_32SC1: 4;
CV_32SC2: 12;
CV_32SC3: 20;
CV_32SC4: 28;
CV_32FC1: 5;
CV_32FC2: 13;
CV_32FC3: 21;
CV_32FC4: 29;
CV_64FC1: 6;
CV_64FC2: 14;
CV_64FC3: 22;
CV_64FC4: 30;
CV_8U: 0;
CV_8S: 1;
CV_16U: 2;
CV_16S: 3;
CV_32S: 4;
CV_32F: 5;
CV_64F: 6;
INT_MIN: -2147483648;
INT_MAX: 2147483647;
ACCESS_FAST: 67108864;
ACCESS_MASK: 50331648;
ACCESS_READ: 16777216;
ACCESS_RW: 50331648;
ACCESS_WRITE: 33554432;
ADAPTIVE_THRESH_GAUSSIAN_C: 1;
ADAPTIVE_THRESH_MEAN_C: 0;
BORDER_CONSTANT: 0;
BORDER_DEFAULT: 4;
BORDER_ISOLATED: 16;
BORDER_REFLECT: 2;
BORDER_REFLECT101: 4;
BORDER_REFLECT_101: 4;
BORDER_REPLICATE: 1;
BORDER_TRANSPARENT: 5;
BORDER_WRAP: 3;
CASCADE_DO_CANNY_PRUNING: 1;
CASCADE_DO_ROUGH_SEARCH: 8;
CASCADE_FIND_BIGGEST_OBJECT: 4;
CASCADE_SCALE_IMAGE: 2;
CCL_DEFAULT: -1;
CCL_GRANA: 1;
CCL_WU: 0;
CC_STAT_AREA: 4;
CC_STAT_HEIGHT: 3;
CC_STAT_LEFT: 0;
CC_STAT_MAX: 5;
CC_STAT_TOP: 1;
CC_STAT_WIDTH: 2;
CHAIN_APPROX_NONE: 1;
CHAIN_APPROX_SIMPLE: 2;
CHAIN_APPROX_TC89_KCOS: 4;
CHAIN_APPROX_TC89_L1: 3;
CMP_EQ: 0;
CMP_GE: 2;
CMP_GT: 1;
CMP_LE: 4;
CMP_LT: 3;
CMP_NE: 5;
COLORMAP_AUTUMN: 0;
COLORMAP_BONE: 1;
COLORMAP_COOL: 8;
COLORMAP_HOT: 11;
COLORMAP_HSV: 9;
COLORMAP_JET: 2;
COLORMAP_OCEAN: 5;
COLORMAP_PARULA: 12;
COLORMAP_PINK: 10;
COLORMAP_RAINBOW: 4;
COLORMAP_SPRING: 7;
COLORMAP_SUMMER: 6;
COLORMAP_WINTER: 3;
COLOR_BGR2BGR555: 22;
COLOR_BGR2BGR565: 12;
COLOR_BGR2BGRA: 0;
COLOR_BGR2GRAY: 6;
COLOR_BGR2HLS: 52;
COLOR_BGR2HLS_FULL: 68;
COLOR_BGR2HSV: 40;
COLOR_BGR2HSV_FULL: 66;
COLOR_BGR2Lab: 44;
COLOR_BGR2Luv: 50;
COLOR_BGR2RGB: 4;
COLOR_BGR2RGBA: 2;
COLOR_BGR2XYZ: 32;
COLOR_BGR2YCrCb: 36;
COLOR_BGR2YUV: 82;
COLOR_BGR2YUV_I420: 128;
COLOR_BGR2YUV_IYUV: 128;
COLOR_BGR2YUV_YV12: 132;
COLOR_BGR5552BGR: 24;
COLOR_BGR5552BGRA: 28;
COLOR_BGR5552GRAY: 31;
COLOR_BGR5552RGB: 25;
COLOR_BGR5552RGBA: 29;
COLOR_BGR5652BGR: 14;
COLOR_BGR5652BGRA: 18;
COLOR_BGR5652GRAY: 21;
COLOR_BGR5652RGB: 15;
COLOR_BGR5652RGBA: 19;
COLOR_BGRA2BGR: 1;
COLOR_BGRA2BGR555: 26;
COLOR_BGRA2BGR565: 16;
COLOR_BGRA2GRAY: 10;
COLOR_BGRA2RGB: 3;
COLOR_BGRA2RGBA: 5;
COLOR_BGRA2YUV_I420: 130;
COLOR_BGRA2YUV_IYUV: 130;
COLOR_BGRA2YUV_YV12: 134;
COLOR_BayerBG2BGR: 46;
COLOR_BayerBG2BGRA: 139;
COLOR_BayerBG2BGR_EA: 135;
COLOR_BayerBG2BGR_VNG: 62;
COLOR_BayerBG2GRAY: 86;
COLOR_BayerBG2RGB: 48;
COLOR_BayerBG2RGBA: 141;
COLOR_BayerBG2RGB_EA: 137;
COLOR_BayerBG2RGB_VNG: 64;
COLOR_BayerGB2BGR: 47;
COLOR_BayerGB2BGRA: 140;
COLOR_BayerGB2BGR_EA: 136;
COLOR_BayerGB2BGR_VNG: 63;
COLOR_BayerGB2GRAY: 87;
COLOR_BayerGB2RGB: 49;
COLOR_BayerGB2RGBA: 142;
COLOR_BayerGB2RGB_EA: 138;
COLOR_BayerGB2RGB_VNG: 65;
COLOR_BayerGR2BGR: 49;
COLOR_BayerGR2BGRA: 142;
COLOR_BayerGR2BGR_EA: 138;
COLOR_BayerGR2BGR_VNG: 65;
COLOR_BayerGR2GRAY: 89;
COLOR_BayerGR2RGB: 47;
COLOR_BayerGR2RGBA: 140;
COLOR_BayerGR2RGB_EA: 136;
COLOR_BayerGR2RGB_VNG: 63;
COLOR_BayerRG2BGR: 48;
COLOR_BayerRG2BGRA: 141;
COLOR_BayerRG2BGR_EA: 137;
COLOR_BayerRG2BGR_VNG: 64;
COLOR_BayerRG2GRAY: 88;
COLOR_BayerRG2RGB: 46;
COLOR_BayerRG2RGBA: 139;
COLOR_BayerRG2RGB_EA: 135;
COLOR_BayerRG2RGB_VNG: 62;
COLOR_COLORCVT_MAX: 143;
COLOR_GRAY2BGR: 8;
COLOR_GRAY2BGR555: 30;
COLOR_GRAY2BGR565: 20;
COLOR_GRAY2BGRA: 9;
COLOR_GRAY2RGB: 8;
COLOR_GRAY2RGBA: 9;
COLOR_HLS2BGR: 60;
COLOR_HLS2BGR_FULL: 72;
COLOR_HLS2RGB: 61;
COLOR_HLS2RGB_FULL: 73;
COLOR_HSV2BGR: 54;
COLOR_HSV2BGR_FULL: 70;
COLOR_HSV2RGB: 55;
COLOR_HSV2RGB_FULL: 71;
COLOR_LBGR2Lab: 74;
COLOR_LBGR2Luv: 76;
COLOR_LRGB2Lab: 75;
COLOR_LRGB2Luv: 77;
COLOR_Lab2BGR: 56;
COLOR_Lab2LBGR: 78;
COLOR_Lab2LRGB: 79;
COLOR_Lab2RGB: 57;
COLOR_Luv2BGR: 58;
COLOR_Luv2LBGR: 80;
COLOR_Luv2LRGB: 81;
COLOR_Luv2RGB: 59;
COLOR_RGB2BGR: 4;
COLOR_RGB2BGR555: 23;
COLOR_RGB2BGR565: 13;
COLOR_RGB2BGRA: 2;
COLOR_RGB2GRAY: 7;
COLOR_RGB2HLS: 53;
COLOR_RGB2HLS_FULL: 69;
COLOR_RGB2HSV: 41;
COLOR_RGB2HSV_FULL: 67;
COLOR_RGB2Lab: 45;
COLOR_RGB2Luv: 51;
COLOR_RGB2RGBA: 0;
COLOR_RGB2XYZ: 33;
COLOR_RGB2YCrCb: 37;
COLOR_RGB2YUV: 83;
COLOR_RGB2YUV_I420: 127;
COLOR_RGB2YUV_IYUV: 127;
COLOR_RGB2YUV_YV12: 131;
COLOR_RGBA2BGR: 3;
COLOR_RGBA2BGR555: 27;
COLOR_RGBA2BGR565: 17;
COLOR_RGBA2BGRA: 5;
COLOR_RGBA2GRAY: 11;
COLOR_RGBA2RGB: 1;
COLOR_RGBA2YUV_I420: 129;
COLOR_RGBA2YUV_IYUV: 129;
COLOR_RGBA2YUV_YV12: 133;
COLOR_RGBA2mRGBA: 125;
COLOR_XYZ2BGR: 34;
COLOR_XYZ2RGB: 35;
COLOR_YCrCb2BGR: 38;
COLOR_YCrCb2RGB: 39;
COLOR_YUV2BGR: 84;
COLOR_YUV2BGRA_I420: 105;
COLOR_YUV2BGRA_IYUV: 105;
COLOR_YUV2BGRA_NV12: 95;
COLOR_YUV2BGRA_NV21: 97;
COLOR_YUV2BGRA_UYNV: 112;
COLOR_YUV2BGRA_UYVY: 112;
COLOR_YUV2BGRA_Y422: 112;
COLOR_YUV2BGRA_YUNV: 120;
COLOR_YUV2BGRA_YUY2: 120;
COLOR_YUV2BGRA_YUYV: 120;
COLOR_YUV2BGRA_YV12: 103;
COLOR_YUV2BGRA_YVYU: 122;
COLOR_YUV2BGR_I420: 101;
COLOR_YUV2BGR_IYUV: 101;
COLOR_YUV2BGR_NV12: 91;
COLOR_YUV2BGR_NV21: 93;
COLOR_YUV2BGR_UYNV: 108;
COLOR_YUV2BGR_UYVY: 108;
COLOR_YUV2BGR_Y422: 108;
COLOR_YUV2BGR_YUNV: 116;
COLOR_YUV2BGR_YUY2: 116;
COLOR_YUV2BGR_YUYV: 116;
COLOR_YUV2BGR_YV12: 99;
COLOR_YUV2BGR_YVYU: 118;
COLOR_YUV2GRAY_420: 106;
COLOR_YUV2GRAY_I420: 106;
COLOR_YUV2GRAY_IYUV: 106;
COLOR_YUV2GRAY_NV12: 106;
COLOR_YUV2GRAY_NV21: 106;
COLOR_YUV2GRAY_UYNV: 123;
COLOR_YUV2GRAY_UYVY: 123;
COLOR_YUV2GRAY_Y422: 123;
COLOR_YUV2GRAY_YUNV: 124;
COLOR_YUV2GRAY_YUY2: 124;
COLOR_YUV2GRAY_YUYV: 124;
COLOR_YUV2GRAY_YV12: 106;
COLOR_YUV2GRAY_YVYU: 124;
COLOR_YUV2RGB: 85;
COLOR_YUV2RGBA_I420: 104;
COLOR_YUV2RGBA_IYUV: 104;
COLOR_YUV2RGBA_NV12: 94;
COLOR_YUV2RGBA_NV21: 96;
COLOR_YUV2RGBA_UYNV: 111;
COLOR_YUV2RGBA_UYVY: 111;
COLOR_YUV2RGBA_Y422: 111;
COLOR_YUV2RGBA_YUNV: 119;
COLOR_YUV2RGBA_YUY2: 119;
COLOR_YUV2RGBA_YUYV: 119;
COLOR_YUV2RGBA_YV12: 102;
COLOR_YUV2RGBA_YVYU: 121;
COLOR_YUV2RGB_I420: 100;
COLOR_YUV2RGB_IYUV: 100;
COLOR_YUV2RGB_NV12: 90;
COLOR_YUV2RGB_NV21: 92;
COLOR_YUV2RGB_UYNV: 107;
COLOR_YUV2RGB_UYVY: 107;
COLOR_YUV2RGB_Y422: 107;
COLOR_YUV2RGB_YUNV: 115;
COLOR_YUV2RGB_YUY2: 115;
COLOR_YUV2RGB_YUYV: 115;
COLOR_YUV2RGB_YV12: 98;
COLOR_YUV2RGB_YVYU: 117;
COLOR_YUV420p2BGR: 99;
COLOR_YUV420p2BGRA: 103;
COLOR_YUV420p2GRAY: 106;
COLOR_YUV420p2RGB: 98;
COLOR_YUV420p2RGBA: 102;
COLOR_YUV420sp2BGR: 93;
COLOR_YUV420sp2BGRA: 97;
COLOR_YUV420sp2GRAY: 106;
COLOR_YUV420sp2RGB: 92;
COLOR_YUV420sp2RGBA: 96;
COLOR_mRGBA2RGBA: 126;
CONTOURS_MATCH_I1: 1;
CONTOURS_MATCH_I2: 2;
CONTOURS_MATCH_I3: 3;
COVAR_COLS: 16;
COVAR_NORMAL: 1;
COVAR_ROWS: 8;
COVAR_SCALE: 4;
COVAR_SCRAMBLED: 0;
COVAR_USE_AVG: 2;
DCT_INVERSE: 1;
DCT_ROWS: 4;
DECOMP_CHOLESKY: 3;
DECOMP_EIG: 2;
DECOMP_LU: 0;
DECOMP_NORMAL: 16;
DECOMP_QR: 4;
DECOMP_SVD: 1;
DFT_COMPLEX_INPUT: 64;
DFT_COMPLEX_OUTPUT: 16;
DFT_INVERSE: 1;
DFT_REAL_OUTPUT: 32;
DFT_ROWS: 4;
DFT_SCALE: 2;
DIST_C: 3;
DIST_FAIR: 5;
DIST_HUBER: 7;
DIST_L1: 1;
DIST_L12: 4;
DIST_L2: 2;
DIST_LABEL_CCOMP: 0;
DIST_LABEL_PIXEL: 1;
DIST_MASK_3: 3;
DIST_MASK_5: 5;
DIST_MASK_PRECISE: 0;
DIST_USER: -1;
DIST_WELSCH: 6;
FILLED: -1;
FLOODFILL_FIXED_RANGE: 65536;
FLOODFILL_MASK_ONLY: 131072;
FONT_HERSHEY_COMPLEX: 3;
FONT_HERSHEY_COMPLEX_SMALL: 5;
FONT_HERSHEY_DUPLEX: 2;
FONT_HERSHEY_PLAIN: 1;
FONT_HERSHEY_SCRIPT_COMPLEX: 7;
FONT_HERSHEY_SCRIPT_SIMPLEX: 6;
FONT_HERSHEY_SIMPLEX: 0;
FONT_HERSHEY_TRIPLEX: 4;
FONT_ITALIC: 16;
FileNode_EMPTY: 32;
FileNode_FLOAT: 2;
FileNode_FLOW: 8;
FileNode_INT: 1;
FileNode_MAP: 6;
FileNode_NAMED: 64;
FileNode_NONE: 0;
FileNode_REAL: 2;
FileNode_REF: 4;
FileNode_SEQ: 5;
FileNode_STR: 3;
FileNode_STRING: 3;
FileNode_TYPE_MASK: 7;
FileNode_USER: 16;
FileStorage_APPEND: 2;
FileStorage_BASE64: 64;
FileStorage_FORMAT_AUTO: 0;
FileStorage_FORMAT_JSON: 24;
FileStorage_FORMAT_MASK: 56;
FileStorage_FORMAT_XML: 8;
FileStorage_FORMAT_YAML: 16;
FileStorage_INSIDE_MAP: 4;
FileStorage_MEMORY: 4;
FileStorage_NAME_EXPECTED: 2;
FileStorage_READ: 0;
FileStorage_UNDEFINED: 0;
FileStorage_VALUE_EXPECTED: 1;
FileStorage_WRITE: 1;
FileStorage_WRITE_BASE64: 65;
Formatter_FMT_C: 5;
Formatter_FMT_CSV: 2;
Formatter_FMT_DEFAULT: 0;
Formatter_FMT_MATLAB: 1;
Formatter_FMT_NUMPY: 4;
Formatter_FMT_PYTHON: 3;
GC_BGD: 0;
GC_EVAL: 2;
GC_FGD: 1;
GC_INIT_WITH_MASK: 1;
GC_INIT_WITH_RECT: 0;
GC_PR_BGD: 2;
GC_PR_FGD: 3;
GEMM_1_T: 1;
GEMM_2_T: 2;
GEMM_3_T: 4;
HISTCMP_BHATTACHARYYA: 3;
HISTCMP_CHISQR: 1;
HISTCMP_CHISQR_ALT: 4;
HISTCMP_CORREL: 0;
HISTCMP_HELLINGER: 3;
HISTCMP_INTERSECT: 2;
HISTCMP_KL_DIV: 5;
HOGDescriptor_DEFAULT_NLEVELS: 64;
HOGDescriptor_L2Hys: 0;
HOUGH_GRADIENT: 3;
HOUGH_MULTI_SCALE: 2;
HOUGH_PROBABILISTIC: 1;
HOUGH_STANDARD: 0;
Hamming_normType: 6;
INTERSECT_FULL: 2;
INTERSECT_NONE: 0;
INTERSECT_PARTIAL: 1;
INTER_AREA: 3;
INTER_BITS: 5;
INTER_BITS2: 10;
INTER_CUBIC: 2;
INTER_LANCZOS4: 4;
INTER_LINEAR: 1;
INTER_MAX: 7;
INTER_NEAREST: 0;
INTER_TAB_SIZE: 32;
INTER_TAB_SIZE2: 1024;
KMEANS_PP_CENTERS: 2;
KMEANS_RANDOM_CENTERS: 0;
KMEANS_USE_INITIAL_LABELS: 1;
LINE_4: 4;
LINE_8: 8;
LINE_AA: 16;
LSD_REFINE_ADV: 2;
LSD_REFINE_NONE: 0;
LSD_REFINE_STD: 1;
MARKER_CROSS: 0;
MARKER_DIAMOND: 3;
MARKER_SQUARE: 4;
MARKER_STAR: 2;
MARKER_TILTED_CROSS: 1;
MARKER_TRIANGLE_DOWN: 6;
MARKER_TRIANGLE_UP: 5;
MORPH_BLACKHAT: 6;
MORPH_CLOSE: 3;
MORPH_CROSS: 1;
MORPH_DILATE: 1;
MORPH_ELLIPSE: 2;
MORPH_ERODE: 0;
MORPH_GRADIENT: 4;
MORPH_HITMISS: 7;
MORPH_OPEN: 2;
MORPH_RECT: 0;
MORPH_TOPHAT: 5;
MOTION_AFFINE: 2;
MOTION_EUCLIDEAN: 1;
MOTION_HOMOGRAPHY: 3;
MOTION_TRANSLATION: 0;
Mat_AUTO_STEP: 0;
Mat_CONTINUOUS_FLAG: 16384;
Mat_MAGIC_VAL: 1124007936;
Mat_SUBMATRIX_FLAG: 32768;
NORM_HAMMING: 6;
NORM_HAMMING2: 7;
NORM_INF: 1;
NORM_L1: 2;
NORM_L2: 4;
NORM_L2SQR: 5;
NORM_MINMAX: 32;
NORM_RELATIVE: 8;
NORM_TYPE_MASK: 7;
OPTFLOW_FARNEBACK_GAUSSIAN: 256;
OPTFLOW_LK_GET_MIN_EIGENVALS: 8;
OPTFLOW_USE_INITIAL_FLOW: 4;
PCA_DATA_AS_COL: 1;
PCA_DATA_AS_ROW: 0;
PCA_USE_AVG: 2;
PROJ_SPHERICAL_EQRECT: 1;
PROJ_SPHERICAL_ORTHO: 0;
Param_ALGORITHM: 6;
Param_BOOLEAN: 1;
Param_FLOAT: 7;
Param_INT: 0;
Param_MAT: 4;
Param_MAT_VECTOR: 5;
Param_REAL: 2;
Param_STRING: 3;
Param_UCHAR: 11;
Param_UINT64: 9;
Param_UNSIGNED_INT: 8;
REDUCE_AVG: 1;
REDUCE_MAX: 2;
REDUCE_MIN: 3;
REDUCE_SUM: 0;
RETR_CCOMP: 2;
RETR_EXTERNAL: 0;
RETR_FLOODFILL: 4;
RETR_LIST: 1;
RETR_TREE: 3;
RNG_NORMAL: 1;
RNG_UNIFORM: 0;
ROTATE_180: 1;
ROTATE_90_CLOCKWISE: 0;
ROTATE_90_COUNTERCLOCKWISE: 2;
SOLVELP_MULTI: 1;
SOLVELP_SINGLE: 0;
SOLVELP_UNBOUNDED: -2;
SOLVELP_UNFEASIBLE: -1;
SORT_ASCENDING: 0;
SORT_DESCENDING: 16;
SORT_EVERY_COLUMN: 1;
SORT_EVERY_ROW: 0;
SVD_FULL_UV: 4;
SVD_MODIFY_A: 1;
SVD_NO_UV: 2;
Subdiv2D_NEXT_AROUND_DST: 34;
Subdiv2D_NEXT_AROUND_LEFT: 19;
Subdiv2D_NEXT_AROUND_ORG: 0;
Subdiv2D_NEXT_AROUND_RIGHT: 49;
Subdiv2D_PREV_AROUND_DST: 51;
Subdiv2D_PREV_AROUND_LEFT: 32;
Subdiv2D_PREV_AROUND_ORG: 17;
Subdiv2D_PREV_AROUND_RIGHT: 2;
Subdiv2D_PTLOC_ERROR: -2;
Subdiv2D_PTLOC_INSIDE: 0;
Subdiv2D_PTLOC_ON_EDGE: 2;
Subdiv2D_PTLOC_OUTSIDE_RECT: -1;
Subdiv2D_PTLOC_VERTEX: 1;
THRESH_BINARY: 0;
THRESH_BINARY_INV: 1;
THRESH_MASK: 7;
THRESH_OTSU: 8;
THRESH_TOZERO: 3;
THRESH_TOZERO_INV: 4;
THRESH_TRIANGLE: 16;
THRESH_TRUNC: 2;
TM_CCOEFF: 4;
TM_CCOEFF_NORMED: 5;
TM_CCORR: 2;
TM_CCORR_NORMED: 3;
TM_SQDIFF: 0;
TM_SQDIFF_NORMED: 1;
TermCriteria_COUNT: 1;
TermCriteria_EPS: 2;
TermCriteria_MAX_ITER: 1;
UMatData_ASYNC_CLEANUP: 128;
UMatData_COPY_ON_MAP: 1;
UMatData_DEVICE_COPY_OBSOLETE: 4;
UMatData_DEVICE_MEM_MAPPED: 64;
UMatData_HOST_COPY_OBSOLETE: 2;
UMatData_TEMP_COPIED_UMAT: 24;
UMatData_TEMP_UMAT: 8;
UMatData_USER_ALLOCATED: 32;
UMat_AUTO_STEP: 0;
UMat_CONTINUOUS_FLAG: 16384;
UMat_MAGIC_VAL: 1124007936;
UMat_SUBMATRIX_FLAG: 32768;
USAGE_ALLOCATE_DEVICE_MEMORY: 2;
USAGE_ALLOCATE_HOST_MEMORY: 1;
USAGE_ALLOCATE_SHARED_MEMORY: 4;
USAGE_DEFAULT: 0;
WARP_FILL_OUTLIERS: 8;
WARP_INVERSE_MAP: 16;
_InputArray_CUDA_GPU_MAT: 589824;
_InputArray_CUDA_HOST_MEM: 524288;
_InputArray_EXPR: 393216;
_InputArray_FIXED_SIZE: 1073741824;
_InputArray_FIXED_TYPE: -2147483648;
_InputArray_KIND_MASK: 2031616;
_InputArray_KIND_SHIFT: 16;
_InputArray_MAT: 65536;
_InputArray_MATX: 131072;
_InputArray_NONE: 0;
_InputArray_OPENGL_BUFFER: 458752;
_InputArray_STD_ARRAY: 917504;
_InputArray_STD_ARRAY_MAT: 983040;
_InputArray_STD_BOOL_VECTOR: 786432;
_InputArray_STD_VECTOR: 196608;
_InputArray_STD_VECTOR_CUDA_GPU_MAT: 851968;
_InputArray_STD_VECTOR_MAT: 327680;
_InputArray_STD_VECTOR_UMAT: 720896;
_InputArray_STD_VECTOR_VECTOR: 262144;
_InputArray_UMAT: 655360;
_OutputArray_DEPTH_MASK_16S: 8;
_OutputArray_DEPTH_MASK_16U: 4;
_OutputArray_DEPTH_MASK_32F: 32;
_OutputArray_DEPTH_MASK_32S: 16;
_OutputArray_DEPTH_MASK_64F: 64;
_OutputArray_DEPTH_MASK_8S: 2;
_OutputArray_DEPTH_MASK_8U: 1;
_OutputArray_DEPTH_MASK_ALL: 127;
_OutputArray_DEPTH_MASK_ALL_BUT_8S: 125;
_OutputArray_DEPTH_MASK_FLT: 96;
__UMAT_USAGE_FLAGS_32BIT: 2147483647;
BadAlign: -21;
BadAlphaChannel: -18;
BadCOI: -24;
BadCallBack: -22;
BadDataPtr: -12;
BadDepth: -17;
BadImageSize: -10;
BadModelOrChSeq: -14;
BadNumChannel1U: -16;
BadNumChannels: -15;
BadOffset: -11;
BadOrder: -19;
BadOrigin: -20;
BadROISize: -25;
BadStep: -13;
BadTileSize: -23;
GpuApiCallError: -217;
GpuNotSupported: -216;
HeaderIsNull: -9;
MaskIsTiled: -26;
OpenCLApiCallError: -220;
OpenCLDoubleNotSupported: -221;
OpenCLInitError: -222;
OpenCLNoAMDBlasFft: -223;
OpenGlApiCallError: -219;
OpenGlNotSupported: -218;
StsAssert: -215;
StsAutoTrace: -8;
StsBackTrace: -1;
StsBadArg: -5;
StsBadFlag: -206;
StsBadFunc: -6;
StsBadMask: -208;
StsBadMemBlock: -214;
StsBadPoint: -207;
StsBadSize: -201;
StsDivByZero: -202;
StsError: -2;
StsFilterOffsetErr: -31;
StsFilterStructContentErr: -29;
StsInplaceNotSupported: -203;
StsInternal: -3;
StsKernelStructContentErr: -30;
StsNoConv: -7;
StsNoMem: -4;
StsNotImplemented: -213;
StsNullPtr: -27;
StsObjectNotFound: -204;
StsOk: 0;
StsOutOfRange: -211;
StsParseError: -212;
StsUnmatchedFormats: -205;
StsUnmatchedSizes: -209;
StsUnsupportedFormat: -210;
StsVecLengthErr: -28;
FLAGS_EXPAND_SAME_NAMES: 2;
FLAGS_MAPPING: 1;
FLAGS_NONE: 0;
IMPL_IPP: 1;
IMPL_OPENCL: 2;
IMPL_PLAIN: 0;
TYPE_FUN: 3;
TYPE_GENERAL: 0;
TYPE_MARKER: 1;
TYPE_WRAPPER: 2;
Mat_DEPTH_MASK: 7;
Mat_MAGIC_MASK: 4294901760;
Mat_TYPE_MASK: 4095;
SparseMat_HASH_BIT: 2147483648;
SparseMat_HASH_SCALE: 1540483477;
SparseMat_MAGIC_VAL: 1123876864;
SparseMat_MAX_DIM: 32;
UMat_DEPTH_MASK: 7;
UMat_MAGIC_MASK: 4294901760;
UMat_TYPE_MASK: 4095;

}