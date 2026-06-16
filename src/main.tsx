import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Matter from "matter-js";
import {
  Camera,
  CirclePlus,
  Crosshair,
  Eraser,
  Crop,
  FlaskConical,
  FlipHorizontal,
  FlipVertical,
  Grid3x3,
  MousePointerClick,
  Pause,
  Pipette,
  Play,
  Projector,
  RotateCcw,
  Save,
  Square,
  SlidersHorizontal,
  Target,
  Upload,
  VideoOff,
} from "lucide-react";
import "./styles.css";

type Point = { x: number; y: number };
type Marker = { id: string; points: Point[]; center: Point; area: number };
type Homography = number[];
type Rect = { x: number; y: number; width: number; height: number };
type Bounds = { minX: number; maxX: number; minY: number; maxY: number };
type Tool = "none" | "calibrate" | "pickColor" | "addBall" | "addCollider" | "detectArea";

const PROJECTOR_WIDTH = 1280;
const PROJECTOR_HEIGHT = 720;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
const BODY_LABEL = "marker-collider";
const MANUAL_BODY_LABEL = "manual-collider";
const BALL_RADIUS = 22;
const TEST_PATTERN_COLORS = ["#ffffff", "#ffff00", "#00ffff", "#00ff00", "#ff00ff", "#ff0000", "#0000ff", "#000000"];
const DEFAULT_COLOR_SETTINGS = { hue: 150, tolerance: 14, saturation: 90, value: 80 };
type ColorSettings = typeof DEFAULT_COLOR_SETTINGS;
const DEMO_MARKERS: Marker[] = [
  {
    id: "demo-left",
    points: [
      { x: 188, y: 286 },
      { x: 418, y: 250 },
      { x: 450, y: 292 },
      { x: 210, y: 338 },
    ],
    center: { x: 316, y: 292 },
    area: 12400,
  },
  {
    id: "demo-right",
    points: [
      { x: 384, y: 306 },
      { x: 524, y: 266 },
      { x: 544, y: 305 },
      { x: 396, y: 354 },
    ],
    center: { x: 462, y: 308 },
    area: 8200,
  },
];

const defaultCameraPoints: Point[] = [
  { x: 40, y: 40 },
  { x: CAMERA_WIDTH - 40, y: 40 },
  { x: CAMERA_WIDTH - 40, y: CAMERA_HEIGHT - 40 },
  { x: 40, y: CAMERA_HEIGHT - 40 },
];

const projectorCorners: Point[] = [
  { x: 0, y: 0 },
  { x: PROJECTOR_WIDTH, y: 0 },
  { x: PROJECTOR_WIDTH, y: PROJECTOR_HEIGHT },
  { x: 0, y: PROJECTOR_HEIGHT },
];

function orderCorners(points: Point[]): Point[] {
  const bySum = [...points].sort((a, b) => a.x + a.y - (b.x + b.y));
  const topLeft = bySum[0];
  const bottomRight = bySum[3];
  const byDiff = [...points].sort((a, b) => a.y - a.x - (b.y - b.x));
  const topRight = byDiff[0];
  const bottomLeft = byDiff[3];
  return [topLeft, topRight, bottomRight, bottomLeft];
}

function getContentRelativePoint(
  canvas: HTMLCanvasElement,
  clientX: number,
  clientY: number,
  contentWidth: number,
  contentHeight: number,
): Point {
  const rect = canvas.getBoundingClientRect();
  const boxAspect = rect.width / rect.height;
  const contentAspect = contentWidth / contentHeight;

  let displayWidth = rect.width;
  let displayHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (boxAspect > contentAspect) {
    displayWidth = rect.height * contentAspect;
    offsetX = (rect.width - displayWidth) / 2;
  } else {
    displayHeight = rect.width / contentAspect;
    offsetY = (rect.height - displayHeight) / 2;
  }

  return {
    x: ((clientX - rect.left - offsetX) / displayWidth) * contentWidth,
    y: ((clientY - rect.top - offsetY) / displayHeight) * contentHeight,
  };
}

function rgbToOpenCvHsv(r: number, g: number, b: number) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta !== 0) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }

  const s = max === 0 ? 0 : delta / max;
  const v = max;

  return { h: Math.round(h / 2), s: Math.round(s * 255), v: Math.round(v * 255) };
}

function pointInQuad(point: Point, quad: Point[]) {
  let inside = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
    const xi = quad[i].x;
    const yi = quad[i].y;
    const xj = quad[j].x;
    const yj = quad[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 0.00001) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function computeHomography(src: Point[], dst: Point[]): Homography {
  const matrix: number[][] = [];
  const vector: number[] = [];

  for (let i = 0; i < 4; i += 1) {
    const { x, y } = src[i];
    const u = dst[i].x;
    const v = dst[i].y;
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    vector.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    vector.push(v);
  }

  const h = solveLinearSystem(matrix, vector);
  return [...h, 1];
}

function solveLinearSystem(matrix: number[][], vector: number[]) {
  const n = vector.length;
  const augmented = matrix.map((row, i) => [...row, vector[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) {
        pivot = row;
      }
    }
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];

    const pivotValue = augmented[col][col] || 1e-12;
    for (let j = col; j <= n; j += 1) {
      augmented[col][j] /= pivotValue;
    }
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = augmented[row][col];
      for (let j = col; j <= n; j += 1) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  return augmented.map((row) => row[n]);
}

function transformPoint(point: Point, h: Homography, offset: Point): Point {
  const denominator = h[6] * point.x + h[7] * point.y + h[8];
  return {
    x: (h[0] * point.x + h[1] * point.y + h[2]) / denominator + offset.x,
    y: (h[3] * point.x + h[4] * point.y + h[5]) / denominator + offset.y,
  };
}

function smoothMarkers(previous: Marker[], next: Marker[]) {
  return next.map((marker) => {
    const closest = previous
      .map((item) => ({
        item,
        distance: Math.hypot(item.center.x - marker.center.x, item.center.y - marker.center.y),
      }))
      .sort((a, b) => a.distance - b.distance)[0];

    if (!closest || closest.distance > 80) return marker;
    const alpha = 0.72;
    return {
      ...marker,
      center: {
        x: closest.item.center.x * alpha + marker.center.x * (1 - alpha),
        y: closest.item.center.y * alpha + marker.center.y * (1 - alpha),
      },
      points: marker.points.map((point, index) => {
        const oldPoint = closest.item.points[index] ?? point;
        return {
          x: oldPoint.x * alpha + point.x * (1 - alpha),
          y: oldPoint.y * alpha + point.y * (1 - alpha),
        };
      }),
    };
  });
}

function detectMarkers(
  cv: any,
  sourceCanvas: HTMLCanvasElement,
  hue: number,
  tolerance: number,
  saturation: number,
  value: number,
  maskCanvas?: HTMLCanvasElement | null,
  roi?: Rect | null,
) {
  const markers: Marker[] = [];
  const src = cv.imread(sourceCanvas);
  const hsv = new cv.Mat();
  const mask = new cv.Mat();
  const hierarchy = new cv.Mat();
  const contours = new cv.MatVector();

  try {
    cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    const lowHue = Math.max(0, hue - tolerance);
    const highHue = Math.min(179, hue + tolerance);
    const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [lowHue, saturation, value, 0]);
    const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [highHue, 255, 255, 255]);
    cv.inRange(hsv, low, high, mask);
    low.delete();
    high.delete();

    const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
    kernel.delete();

    if (roi) {
      const black = new cv.Scalar(0, 0, 0, 0);
      if (roi.y > 0) cv.rectangle(mask, new cv.Point(0, 0), new cv.Point(mask.cols, roi.y), black, -1);
      if (roi.y + roi.height < mask.rows) {
        cv.rectangle(mask, new cv.Point(0, roi.y + roi.height), new cv.Point(mask.cols, mask.rows), black, -1);
      }
      if (roi.x > 0) cv.rectangle(mask, new cv.Point(0, roi.y), new cv.Point(roi.x, roi.y + roi.height), black, -1);
      if (roi.x + roi.width < mask.cols) {
        cv.rectangle(mask, new cv.Point(roi.x + roi.width, roi.y), new cv.Point(mask.cols, roi.y + roi.height), black, -1);
      }
    }

    if (maskCanvas) {
      cv.imshow(maskCanvas, mask);
    }

    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i += 1) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < 350) {
        contour.delete();
        continue;
      }

      const perimeter = cv.arcLength(contour, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);
      const rect = cv.boundingRect(approx);
      const points: Point[] = [];
      for (let row = 0; row < approx.rows; row += 1) {
        points.push({ x: approx.intPtr(row, 0)[0], y: approx.intPtr(row, 0)[1] });
      }

      if (points.length >= 3) {
        markers.push({
          id: `marker-${i}`,
          points,
          center: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
          area,
        });
      }
      approx.delete();
      contour.delete();
    }
  } finally {
    src.delete();
    hsv.delete();
    mask.delete();
    hierarchy.delete();
    contours.delete();
  }

  return markers.sort((a, b) => b.area - a.area).slice(0, 8);
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const projectorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<Matter.Engine | null>(null);
  const leftWallRef = useRef<Matter.Body | null>(null);
  const rightWallRef = useRef<Matter.Body | null>(null);
  const frameRef = useRef(0);
  const previousMarkersRef = useRef<Marker[]>([]);
  const outputWindowRef = useRef<Window | null>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [cvReady, setCvReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [running, setRunning] = useState(true);
  const [outputWindowOpen, setOutputWindowOpen] = useState(false);
  const [flipX, setFlipX] = useState(false);
  const [flipY, setFlipY] = useState(false);
  const [testPattern, setTestPattern] = useState(false);
  const [tool, setTool] = useState<Tool>("none");
  const [cameraRoi, setCameraRoi] = useState<Rect | null>(null);
  const [detectAreaStart, setDetectAreaStart] = useState<Point | null>(null);
  const [cameraPoints, setCameraPoints] = useState<Point[]>(defaultCameraPoints);
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState("OpenCV.jsを読み込み中");
  const colorFileInputRef = useRef<HTMLInputElement | null>(null);
  const [hue, setHue] = useState(DEFAULT_COLOR_SETTINGS.hue);
  const [tolerance, setTolerance] = useState(DEFAULT_COLOR_SETTINGS.tolerance);
  const [saturation, setSaturation] = useState(DEFAULT_COLOR_SETTINGS.saturation);
  const [value, setValue] = useState(DEFAULT_COLOR_SETTINGS.value);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const homography = useMemo(
    () => computeHomography(cameraPoints.length === 4 ? cameraPoints : defaultCameraPoints, projectorCorners),
    [cameraPoints],
  );
  const playAreaBounds = useMemo(() => {
    const roi = cameraRoi ?? { x: 0, y: 0, width: CAMERA_WIDTH, height: CAMERA_HEIGHT };
    const corners = [
      { x: roi.x, y: roi.y },
      { x: roi.x + roi.width, y: roi.y },
      { x: roi.x + roi.width, y: roi.y + roi.height },
      { x: roi.x, y: roi.y + roi.height },
    ].map((point) => transformPoint(point, homography, offset));
    const xs = corners.map((point) => point.x);
    const ys = corners.map((point) => point.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [cameraRoi, homography, offset]);

  useEffect(() => {
    let cancelled = false;
    const waitForCv = () => {
      if (cancelled) return;
      if (window.cv?.Mat) {
        setCvReady(true);
        setStatus("カメラ開始待ち");
        return;
      }
      window.setTimeout(waitForCv, 120);
    };
    waitForCv();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const engine = Matter.Engine.create();
    engine.gravity.y = 0.95;
    const world = engine.world;
    const leftWall = Matter.Bodies.rectangle(-24, PROJECTOR_HEIGHT / 2, 48, PROJECTOR_HEIGHT, {
      isStatic: true,
      label: "left-wall",
    });
    const rightWall = Matter.Bodies.rectangle(PROJECTOR_WIDTH + 24, PROJECTOR_HEIGHT / 2, 48, PROJECTOR_HEIGHT, {
      isStatic: true,
      label: "right-wall",
    });
    const ball = Matter.Bodies.circle(220, 100, BALL_RADIUS, {
      restitution: 0.94,
      friction: 0.02,
      frictionAir: 0.002,
      label: "ball",
    });
    Matter.World.add(world, [leftWall, rightWall, ball]);
    engineRef.current = engine;
    leftWallRef.current = leftWall;
    rightWallRef.current = rightWall;

    return () => {
      Matter.Engine.clear(engine);
      engineRef.current = null;
      leftWallRef.current = null;
      rightWallRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!leftWallRef.current || !rightWallRef.current) return;
    Matter.Body.setPosition(leftWallRef.current, { x: playAreaBounds.minX - 24, y: PROJECTOR_HEIGHT / 2 });
    Matter.Body.setPosition(rightWallRef.current, { x: playAreaBounds.maxX + 24, y: PROJECTOR_HEIGHT / 2 });
  }, [playAreaBounds]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    const startCamera = async () => {
      if (!cvReady) return;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: CAMERA_WIDTH, height: CAMERA_HEIGHT, facingMode: "environment" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setCameraReady(true);
        setDemoMode(false);
        setStatus("カメラ入力を解析中");
      } catch {
        setDemoMode(true);
        setStatus("カメラ未接続: デモモードで動作中");
      }
    };
    startCamera();
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [cvReady]);

  useEffect(() => {
    const cameraCanvas = cameraCanvasRef.current;
    const projectorCanvas = projectorCanvasRef.current;
    if (!cameraCanvas || !projectorCanvas) return;
    const cameraCtx = cameraCanvas.getContext("2d", { willReadFrequently: true });
    const projectorCtx = projectorCanvas.getContext("2d");
    if (!cameraCtx || !projectorCtx) return;

    let lastTime = performance.now();
    let frames = 0;
    let fpsStart = performance.now();

    const renderOutputWindow = () => {
      const outputCanvas = outputCanvasRef.current;
      const outputWindow = outputWindowRef.current;
      if (!outputCanvas || !outputWindow || outputWindow.closed) return;
      const outCtx = outputCanvas.getContext("2d");
      if (!outCtx) return;
      if (testPattern) {
        drawTestPattern(outCtx);
      } else {
        drawProjectorBallsOnly(outCtx, engineRef.current, flipX, flipY);
      }
      if (tool === "calibrate") {
        drawCalibrationMarkers(outCtx, flipX, flipY);
      }
    };

    const loop = (time: number) => {
      frameRef.current = requestAnimationFrame(loop);
      if (!running) {
        drawProjector(projectorCtx, engineRef.current, markers, homography, offset, tool, cameraPoints, cameraRoi);
        renderOutputWindow();
        return;
      }

      const delta = Math.min(16, time - lastTime);
      lastTime = time;
      drawCameraInput(cameraCtx, videoRef.current, demoMode, time);

      let currentMarkers = previousMarkersRef.current;
      if (cvReady && (cameraReady || demoMode)) {
        currentMarkers = demoMode
          ? animatedDemoMarkers(time)
          : detectMarkers(window.cv, cameraCanvas, hue, tolerance, saturation, value, maskCanvasRef.current, cameraRoi);
        currentMarkers = smoothMarkers(previousMarkersRef.current, currentMarkers);
        previousMarkersRef.current = currentMarkers;
      }

      updateColliderBodies(engineRef.current, currentMarkers, homography, offset);
      if (engineRef.current) {
        Matter.Engine.update(engineRef.current, delta);
        removeOffscreenBalls(engineRef.current, playAreaBounds);
      }
      drawProjector(projectorCtx, engineRef.current, currentMarkers, homography, offset, tool, cameraPoints, cameraRoi);
      drawCameraOverlay(cameraCtx, currentMarkers, cameraPoints, tool, cameraRoi, detectAreaStart);
      renderOutputWindow();

      frames += 1;
      if (time - fpsStart > 500) {
        setFps(Math.round((frames * 1000) / (time - fpsStart)));
        setMarkers(currentMarkers);
        frames = 0;
        fpsStart = time;
      }
    };

    frameRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frameRef.current);
  }, [
    cameraPoints,
    cameraReady,
    cameraRoi,
    tool,
    cvReady,
    demoMode,
    detectAreaStart,
    flipX,
    flipY,
    homography,
    hue,
    offset,
    playAreaBounds,
    running,
    saturation,
    testPattern,
    tolerance,
    value,
  ]);

  useEffect(() => {
    if (!demoMode) return;
    const ctx = maskCanvasRef.current?.getContext("2d");
    ctx?.clearRect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
  }, [demoMode]);

  useEffect(() => {
    return () => {
      outputWindowRef.current?.close();
      outputWindowRef.current = null;
      outputCanvasRef.current = null;
    };
  }, []);

  const openProjectorWindow = () => {
    if (outputWindowRef.current && !outputWindowRef.current.closed) {
      outputWindowRef.current.focus();
      return;
    }

    const win = window.open("", "projector-output", `width=${PROJECTOR_WIDTH},height=${PROJECTOR_HEIGHT}`);
    if (!win) return;

    win.document.title = "Projector Output";
    win.document.body.innerHTML = "";
    win.document.body.style.margin = "0";
    win.document.body.style.background = "#000";
    win.document.body.style.overflow = "hidden";
    win.document.body.style.display = "flex";
    win.document.body.style.alignItems = "center";
    win.document.body.style.justifyContent = "center";
    win.document.body.style.height = "100vh";

    const canvas = win.document.createElement("canvas");
    canvas.width = PROJECTOR_WIDTH;
    canvas.height = PROJECTOR_HEIGHT;
    canvas.style.display = "block";
    canvas.style.cursor = "pointer";
    canvas.title = "クリックでフルスクリーン";
    canvas.addEventListener("click", () => {
      canvas.requestFullscreen?.().catch(() => {});
    });
    win.document.body.appendChild(canvas);

    const fitCanvas = () => {
      const scale = Math.min(win.innerWidth / PROJECTOR_WIDTH, win.innerHeight / PROJECTOR_HEIGHT);
      canvas.style.width = `${PROJECTOR_WIDTH * scale}px`;
      canvas.style.height = `${PROJECTOR_HEIGHT * scale}px`;
    };
    fitCanvas();
    win.addEventListener("resize", fitCanvas);
    win.document.addEventListener("fullscreenchange", fitCanvas);

    win.addEventListener("beforeunload", () => {
      outputWindowRef.current = null;
      outputCanvasRef.current = null;
      setOutputWindowOpen(false);
    });

    outputWindowRef.current = win;
    outputCanvasRef.current = canvas;
    setOutputWindowOpen(true);
  };

  const closeProjectorWindow = () => {
    outputWindowRef.current?.close();
    outputWindowRef.current = null;
    outputCanvasRef.current = null;
    setOutputWindowOpen(false);
  };

  const addBallAt = (point: Point) => {
    if (!engineRef.current) return;
    const ball = Matter.Bodies.circle(point.x, point.y, BALL_RADIUS, {
      restitution: 0.94,
      friction: 0.02,
      frictionAir: 0.002,
      label: "ball",
    });
    Matter.World.add(engineRef.current.world, ball);
  };

  const addColliderAt = (point: Point) => {
    if (!engineRef.current) return;
    const collider = Matter.Bodies.rectangle(point.x, point.y, 110, 34, {
      isStatic: true,
      restitution: 0.86,
      friction: 0.04,
      label: MANUAL_BODY_LABEL,
    });
    Matter.World.add(engineRef.current.world, collider);
  };

  const clearManualColliders = () => {
    if (!engineRef.current) return;
    const bodies = Matter.Composite.allBodies(engineRef.current.world).filter(
      (body) => body.label === MANUAL_BODY_LABEL,
    );
    Matter.World.remove(engineRef.current.world, bodies);
  };

  const toggleTool = (next: Tool) => {
    setDetectAreaStart(null);
    setTool((current) => (current === next ? "none" : next));
  };

  const resetCameraRoi = () => {
    setDetectAreaStart(null);
    setCameraRoi(null);
  };

  const resetColorSettings = () => {
    setHue(DEFAULT_COLOR_SETTINGS.hue);
    setTolerance(DEFAULT_COLOR_SETTINGS.tolerance);
    setSaturation(DEFAULT_COLOR_SETTINGS.saturation);
    setValue(DEFAULT_COLOR_SETTINGS.value);
  };

  const saveColorSettings = () => {
    const settings: ColorSettings = { hue, tolerance, saturation, value };
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "color-settings.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const loadColorSettings = () => {
    colorFileInputRef.current?.click();
  };

  const handleColorFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    file
      .text()
      .then((text) => {
        const parsed = JSON.parse(text);
        if (
          typeof parsed.hue !== "number" ||
          typeof parsed.tolerance !== "number" ||
          typeof parsed.saturation !== "number" ||
          typeof parsed.value !== "number"
        ) {
          throw new Error("invalid format");
        }
        setHue(Math.min(179, Math.max(0, parsed.hue)));
        setTolerance(Math.min(45, Math.max(4, parsed.tolerance)));
        setSaturation(Math.min(255, Math.max(0, parsed.saturation)));
        setValue(Math.min(255, Math.max(0, parsed.value)));
        setStatus("色検出設定を読み込みました");
      })
      .catch(() => {
        setStatus("色検出設定の読み込みに失敗しました");
      });
  };

  const handleCameraClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool === "calibrate") {
      const point = getContentRelativePoint(
        event.currentTarget,
        event.clientX,
        event.clientY,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
      );
      setCameraPoints((points) => {
        const next = points.length >= 4 ? [point] : [...points, point];
        if (next.length === 4) {
          setTool("none");
          return orderCorners(next);
        }
        return next;
      });
      return;
    }

    if (tool === "pickColor") {
      const point = getContentRelativePoint(
        event.currentTarget,
        event.clientX,
        event.clientY,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
      );
      const ctx = event.currentTarget.getContext("2d");
      if (ctx) {
        const px = Math.min(CAMERA_WIDTH - 1, Math.max(0, Math.round(point.x)));
        const py = Math.min(CAMERA_HEIGHT - 1, Math.max(0, Math.round(point.y)));
        const [r, g, b] = ctx.getImageData(px, py, 1, 1).data;
        const { h, s, v } = rgbToOpenCvHsv(r, g, b);
        setHue(h);
        setSaturation(Math.max(0, s - 40));
        setValue(Math.max(0, v - 50));
        setStatus(`色を検出しました: H${h} S${s} V${v}`);
      }
      setTool("none");
      return;
    }

    if (tool === "detectArea") {
      const point = getContentRelativePoint(
        event.currentTarget,
        event.clientX,
        event.clientY,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
      );
      if (!detectAreaStart) {
        setDetectAreaStart(point);
        return;
      }
      const rect: Rect = {
        x: Math.min(detectAreaStart.x, point.x),
        y: Math.min(detectAreaStart.y, point.y),
        width: Math.abs(point.x - detectAreaStart.x),
        height: Math.abs(point.y - detectAreaStart.y),
      };
      setDetectAreaStart(null);
      setTool("none");
      if (rect.width > 20 && rect.height > 20) {
        setCameraRoi(rect);
      }
    }
  };

  const handleProjectorClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool !== "addBall" && tool !== "addCollider") return;
    const point = getContentRelativePoint(
      event.currentTarget,
      event.clientX,
      event.clientY,
      PROJECTOR_WIDTH,
      PROJECTOR_HEIGHT,
    );
    if (tool === "addBall") addBallAt(point);
    else addColliderAt(point);
  };

  return (
    <main className="app-shell">
      <section className="control-panel">
        <div className="brand-row">
          <Target size={26} />
          <div>
            <h1>Projection Mapping Lab</h1>
            <p>Camera to collider testbed</p>
          </div>
        </div>

        <div className="status-strip">
          <span className={cvReady ? "dot ready" : "dot"} />
          <span>{status}</span>
        </div>

        <div className="button-grid">
          <button type="button" onClick={() => setRunning((value) => !value)}>
            {running ? <Pause size={18} /> : <Play size={18} />}
            {running ? "停止" : "再開"}
          </button>
          <button type="button" onClick={() => setDemoMode((value) => !value)}>
            {demoMode ? <Camera size={18} /> : <FlaskConical size={18} />}
            {demoMode ? "カメラ" : "デモ"}
          </button>
          <button
            type="button"
            className={outputWindowOpen ? "active" : ""}
            onClick={outputWindowOpen ? closeProjectorWindow : openProjectorWindow}
          >
            <Projector size={18} />
            {outputWindowOpen ? "投影ウィンドウを閉じる" : "プロジェクターへ出力"}
          </button>
          <button type="button" className={flipX ? "active" : ""} onClick={() => setFlipX((value) => !value)}>
            <FlipHorizontal size={18} />
            左右反転
          </button>
          <button type="button" className={flipY ? "active" : ""} onClick={() => setFlipY((value) => !value)}>
            <FlipVertical size={18} />
            上下反転
          </button>
          <button
            type="button"
            className={testPattern ? "active" : ""}
            onClick={() => setTestPattern((value) => !value)}
          >
            <Grid3x3 size={18} />
            テストパターン
          </button>
          <button type="button" onClick={() => addBallAt({ x: 180 + Math.random() * 240, y: 40 })}>
            <CirclePlus size={18} />
            ボール追加
          </button>
          <button
            type="button"
            className={tool === "calibrate" ? "active" : ""}
            onClick={() => {
              setCameraPoints([]);
              setTool("calibrate");
            }}
          >
            <Crosshair size={18} />
            4点補正
          </button>
          <button type="button" onClick={() => setCameraPoints(defaultCameraPoints)}>
            <RotateCcw size={18} />
            補正初期化
          </button>
          <button
            type="button"
            className={tool === "detectArea" ? "active" : ""}
            onClick={() => toggleTool("detectArea")}
          >
            <Crop size={18} />
            判定範囲を設定
          </button>
          <button type="button" onClick={resetCameraRoi}>
            <RotateCcw size={18} />
            判定範囲初期化
          </button>
          <button
            type="button"
            className={tool === "pickColor" ? "active" : ""}
            onClick={() => toggleTool("pickColor")}
          >
            <Pipette size={18} />
            スポイトで色を選択
          </button>
          <button
            type="button"
            className={tool === "addBall" ? "active" : ""}
            onClick={() => toggleTool("addBall")}
          >
            <MousePointerClick size={18} />
            クリックでボール配置
          </button>
          <button
            type="button"
            className={tool === "addCollider" ? "active" : ""}
            onClick={() => toggleTool("addCollider")}
          >
            <Square size={18} />
            クリックでコリジョン配置
          </button>
          <button type="button" onClick={clearManualColliders}>
            <Eraser size={18} />
            コリジョン消去
          </button>
        </div>

        <div className="meter-grid">
          <Metric label="FPS" value={fps} />
          <Metric label="検出数" value={markers.length} />
          <Metric label="補正点" value={`${cameraPoints.length}/4`} />
          <Metric label="入力" value={demoMode ? "DEMO" : cameraReady ? "CAM" : "OFF"} />
        </div>

        <ControlGroup title="色検出">
          <Range label="Hue" min={0} max={179} value={hue} onChange={setHue} />
          <Range label="許容幅" min={4} max={45} value={tolerance} onChange={setTolerance} />
          <Range label="彩度" min={0} max={255} value={saturation} onChange={setSaturation} />
          <Range label="明度" min={0} max={255} value={value} onChange={setValue} />
          <div className="mask-preview-row">
            <span>検出マスク (白=検出範囲)</span>
            <canvas ref={maskCanvasRef} width={CAMERA_WIDTH} height={CAMERA_HEIGHT} className="mask-preview" />
          </div>
          <div className="button-row">
            <button type="button" onClick={resetColorSettings}>
              <RotateCcw size={16} />
              初期値に戻す
            </button>
            <button type="button" onClick={saveColorSettings}>
              <Save size={16} />
              保存
            </button>
            <button type="button" onClick={loadColorSettings}>
              <Upload size={16} />
              読み込み
            </button>
            <input
              ref={colorFileInputRef}
              type="file"
              accept="application/json"
              onChange={handleColorFileChange}
              className="hidden-file-input"
            />
          </div>
        </ControlGroup>

        <ControlGroup title="投影オフセット">
          <Range label="X" min={-160} max={160} value={offset.x} onChange={(x) => setOffset((point) => ({ ...point, x }))} />
          <Range label="Y" min={-120} max={120} value={offset.y} onChange={(y) => setOffset((point) => ({ ...point, y }))} />
        </ControlGroup>
      </section>

      <section className="workbench">
        <div className="stage-toolbar">
          <div>
            <h2>Projector Output</h2>
            <p>
              {tool === "calibrate"
                ? "カメラ映像の四隅を順にクリック"
                : tool === "pickColor"
                  ? "カメラ映像内で検出したい色をクリック"
                  : tool === "addBall"
                    ? "投影画面をクリックしてボールを配置"
                    : tool === "addCollider"
                      ? "投影画面をクリックしてコリジョンを配置"
                      : tool === "detectArea"
                        ? detectAreaStart
                          ? "カメラ映像内でもう一方の角をクリック"
                          : "カメラ映像内で解析したい範囲の対角2点をクリック(1点目)"
                        : "検出した物体が静的コリジョンになります"}
            </p>
          </div>
          <SlidersHorizontal size={22} />
        </div>
        <div className="canvas-grid">
          <div className="canvas-block projector-block">
            <canvas
              ref={projectorCanvasRef}
              width={PROJECTOR_WIDTH}
              height={PROJECTOR_HEIGHT}
              onClick={handleProjectorClick}
              className={tool === "addBall" || tool === "addCollider" ? "is-targeting" : ""}
            />
          </div>
          <div className="canvas-block camera-block">
            <canvas
              ref={cameraCanvasRef}
              width={CAMERA_WIDTH}
              height={CAMERA_HEIGHT}
              onClick={handleCameraClick}
              className={tool === "calibrate" || tool === "pickColor" || tool === "detectArea" ? "is-targeting" : ""}
            />
            <video ref={videoRef} muted playsInline />
            {!cameraReady && !demoMode ? (
              <div className="camera-empty">
                <VideoOff size={24} />
                カメラ待機中
              </div>
            ) : null}
          </div>
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ControlGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="control-group">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function Range({
  label,
  min,
  max,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-row">
      <span>{label}</span>
      <input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <output>{Math.round(value)}</output>
    </label>
  );
}

function drawCameraInput(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement | null,
  demoMode: boolean,
  time: number,
) {
  ctx.clearRect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
  if (!demoMode && video?.readyState && video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
    return;
  }

  const gradient = ctx.createLinearGradient(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
  gradient.addColorStop(0, "#151b26");
  gradient.addColorStop(1, "#262f3d");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  for (let x = 0; x < CAMERA_WIDTH; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, CAMERA_HEIGHT);
    ctx.stroke();
  }
  for (let y = 0; y < CAMERA_HEIGHT; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(CAMERA_WIDTH, y);
    ctx.stroke();
  }

  const markers = animatedDemoMarkers(time);
  ctx.fillStyle = "rgb(255, 0, 210)";
  markers.forEach((marker) => {
    ctx.beginPath();
    marker.points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.fill();
  });
}

function animatedDemoMarkers(time: number): Marker[] {
  const wobble = Math.sin(time / 850) * 18;
  return DEMO_MARKERS.map((marker, index) => ({
    ...marker,
    id: `${marker.id}-${index}`,
    center: { x: marker.center.x + wobble * (index ? -0.6 : 0.8), y: marker.center.y + Math.cos(time / 1200) * 7 },
    points: marker.points.map((point) => ({
      x: point.x + wobble * (index ? -0.6 : 0.8),
      y: point.y + Math.cos(time / 1200) * 7,
    })),
  }));
}

function drawCameraOverlay(
  ctx: CanvasRenderingContext2D,
  markers: Marker[],
  cameraPoints: Point[],
  tool: Tool,
  cameraRoi: Rect | null,
  pendingDetectPoint: Point | null,
) {
  const calibrating = tool === "calibrate";
  markers.forEach((marker) => {
    ctx.strokeStyle = "#37f2a2";
    ctx.lineWidth = 3;
    ctx.beginPath();
    marker.points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(marker.center.x - 3, marker.center.y - 3, 6, 6);
  });

  cameraPoints.forEach((point, index) => {
    ctx.fillStyle = calibrating ? "#ffdf63" : "#5aa7ff";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#111621";
    ctx.font = "12px system-ui";
    ctx.fillText(String(index + 1), point.x - 4, point.y + 4);
  });

  if (cameraRoi) {
    ctx.save();
    ctx.strokeStyle = "#37e0ff";
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 2;
    ctx.strokeRect(cameraRoi.x, cameraRoi.y, cameraRoi.width, cameraRoi.height);
    ctx.restore();
  }

  if (pendingDetectPoint) {
    ctx.fillStyle = "#37e0ff";
    ctx.beginPath();
    ctx.arc(pendingDetectPoint.x, pendingDetectPoint.y, 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateColliderBodies(engine: Matter.Engine | null, markers: Marker[], homography: Homography, offset: Point) {
  if (!engine) return;

  const bodies = Matter.Composite.allBodies(engine.world).filter((body) => body.label === BODY_LABEL);
  Matter.World.remove(engine.world, bodies);

  markers.forEach((marker) => {
    const transformed = marker.points.map((point) => transformPoint(point, homography, offset));
    if (transformed.length < 3) return;
    const center = transformed.reduce(
      (acc, point) => ({ x: acc.x + point.x / transformed.length, y: acc.y + point.y / transformed.length }),
      { x: 0, y: 0 },
    );
    const vertices = transformed.map((point) => ({ x: point.x - center.x, y: point.y - center.y }));
    const body = Matter.Bodies.fromVertices(center.x, center.y, [vertices], {
      isStatic: true,
      restitution: 0.86,
      friction: 0.04,
      label: BODY_LABEL,
    });
    Matter.World.add(engine.world, body);
  });
}

function drawProjector(
  ctx: CanvasRenderingContext2D,
  engine: Matter.Engine | null,
  markers: Marker[],
  homography: Homography,
  offset: Point,
  tool: Tool,
  cameraPoints: Point[],
  cameraRoi: Rect | null,
) {
  const calibrating = tool === "calibrate";
  ctx.clearRect(0, 0, PROJECTOR_WIDTH, PROJECTOR_HEIGHT);
  ctx.fillStyle = "#0e1219";
  ctx.fillRect(0, 0, PROJECTOR_WIDTH, PROJECTOR_HEIGHT);

  const grid = 80;
  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= PROJECTOR_WIDTH; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, PROJECTOR_HEIGHT);
    ctx.stroke();
  }
  for (let y = 0; y <= PROJECTOR_HEIGHT; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(PROJECTOR_WIDTH, y);
    ctx.stroke();
  }

  if (calibrating || cameraPoints.length < 4) {
    projectorCorners.forEach((point, index) => {
      ctx.fillStyle = "#ffdf63";
      ctx.beginPath();
      ctx.arc(point.x || 24, point.y || 24, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#10151f";
      ctx.font = "bold 15px system-ui";
      ctx.fillText(String(index + 1), (point.x || 24) - 4, (point.y || 24) + 5);
    });
  }

  if (cameraRoi) {
    const roiCorners = [
      { x: cameraRoi.x, y: cameraRoi.y },
      { x: cameraRoi.x + cameraRoi.width, y: cameraRoi.y },
      { x: cameraRoi.x + cameraRoi.width, y: cameraRoi.y + cameraRoi.height },
      { x: cameraRoi.x, y: cameraRoi.y + cameraRoi.height },
    ].map((point) => transformPoint(point, homography, offset));
    ctx.save();
    ctx.strokeStyle = "#37e0ff";
    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    roiCorners.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  markers.forEach((marker) => {
    const transformed = marker.points.map((point) => transformPoint(point, homography, offset));
    ctx.fillStyle = "rgba(55, 242, 162, 0.16)";
    ctx.strokeStyle = "#37f2a2";
    ctx.lineWidth = 3;
    ctx.beginPath();
    transformed.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });

  if (!engine) return;
  const bodies = Matter.Composite.allBodies(engine.world);
  bodies.forEach((body) => {
    if (body.label === "ball") {
      drawBall(ctx, body);
    } else if (body.label === MANUAL_BODY_LABEL) {
      ctx.fillStyle = "rgba(255, 176, 59, 0.18)";
      ctx.strokeStyle = "#ffb03b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      body.vertices.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  });
}

function drawBall(ctx: CanvasRenderingContext2D, body: Matter.Body) {
  ctx.fillStyle = "#5aa7ff";
  ctx.beginPath();
  ctx.arc(body.position.x, body.position.y, BALL_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  ctx.stroke();
}

function drawTestPattern(ctx: CanvasRenderingContext2D) {
  ctx.clearRect(0, 0, PROJECTOR_WIDTH, PROJECTOR_HEIGHT);

  const barHeight = PROJECTOR_HEIGHT * 0.6;
  const barWidth = PROJECTOR_WIDTH / TEST_PATTERN_COLORS.length;
  TEST_PATTERN_COLORS.forEach((color, index) => {
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(index * barWidth), 0, Math.ceil(barWidth), barHeight);
  });

  const gridSize = 60;
  for (let y = barHeight; y < PROJECTOR_HEIGHT; y += gridSize) {
    for (let x = 0; x < PROJECTOR_WIDTH; x += gridSize) {
      const isEven = (Math.floor(x / gridSize) + Math.floor((y - barHeight) / gridSize)) % 2 === 0;
      ctx.fillStyle = isEven ? "#ffffff" : "#000000";
      ctx.fillRect(x, y, gridSize, Math.min(gridSize, PROJECTOR_HEIGHT - y));
    }
  }

  ctx.strokeStyle = "#ff0000";
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, PROJECTOR_WIDTH - 4, PROJECTOR_HEIGHT - 4);
}

function drawCalibrationMarkers(ctx: CanvasRenderingContext2D, flipX: boolean, flipY: boolean) {
  projectorCorners.forEach((point, index) => {
    const px = (flipX ? PROJECTOR_WIDTH - point.x : point.x) || 24;
    const py = (flipY ? PROJECTOR_HEIGHT - point.y : point.y) || 24;
    ctx.fillStyle = "#ffdf63";
    ctx.beginPath();
    ctx.arc(px, py, 13, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#10151f";
    ctx.font = "bold 15px system-ui";
    ctx.fillText(String(index + 1), px - 4, py + 5);
  });
}

function drawProjectorBallsOnly(
  ctx: CanvasRenderingContext2D,
  engine: Matter.Engine | null,
  flipX: boolean,
  flipY: boolean,
) {
  ctx.clearRect(0, 0, PROJECTOR_WIDTH, PROJECTOR_HEIGHT);
  if (!engine) return;

  ctx.save();
  ctx.translate(flipX ? PROJECTOR_WIDTH : 0, flipY ? PROJECTOR_HEIGHT : 0);
  ctx.scale(flipX ? -1 : 1, flipY ? -1 : 1);
  Matter.Composite.allBodies(engine.world).forEach((body) => {
    if (body.label === "ball") drawBall(ctx, body);
  });
  ctx.restore();
}

function removeOffscreenBalls(engine: Matter.Engine | null, bounds: Bounds) {
  if (!engine) return;
  const balls = Matter.Composite.allBodies(engine.world).filter((body) => body.label === "ball");
  const offscreen = balls.filter(
    (ball) =>
      ball.position.y - BALL_RADIUS > bounds.maxY ||
      ball.position.x < bounds.minX - BALL_RADIUS ||
      ball.position.x > bounds.maxX + BALL_RADIUS,
  );
  if (offscreen.length > 0) {
    Matter.World.remove(engine.world, offscreen);
  }
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
