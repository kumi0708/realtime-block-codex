import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Matter from "matter-js";
import {
  Camera,
  CirclePlus,
  Crosshair,
  FlaskConical,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Target,
  VideoOff,
} from "lucide-react";
import "./styles.css";

type Point = { x: number; y: number };
type Marker = { id: string; points: Point[]; center: Point; area: number };
type Homography = number[];

const PROJECTOR_WIDTH = 1280;
const PROJECTOR_HEIGHT = 720;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
const BODY_LABEL = "marker-collider";
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
  const engineRef = useRef<Matter.Engine | null>(null);
  const ballRef = useRef<Matter.Body | null>(null);
  const frameRef = useRef(0);
  const previousMarkersRef = useRef<Marker[]>([]);
  const [cvReady, setCvReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [running, setRunning] = useState(true);
  const [calibrating, setCalibrating] = useState(false);
  const [cameraPoints, setCameraPoints] = useState<Point[]>(defaultCameraPoints);
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState("OpenCV.jsを読み込み中");
  const [hue, setHue] = useState(150);
  const [tolerance, setTolerance] = useState(14);
  const [saturation, setSaturation] = useState(90);
  const [value, setValue] = useState(80);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const homography = useMemo(() => computeHomography(cameraPoints, projectorCorners), [cameraPoints]);

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
    const floor = Matter.Bodies.rectangle(PROJECTOR_WIDTH / 2, PROJECTOR_HEIGHT + 26, PROJECTOR_WIDTH, 52, {
      isStatic: true,
      label: "floor",
      render: { fillStyle: "#2f3747" },
    });
    const leftWall = Matter.Bodies.rectangle(-24, PROJECTOR_HEIGHT / 2, 48, PROJECTOR_HEIGHT, {
      isStatic: true,
      label: "left-wall",
    });
    const rightWall = Matter.Bodies.rectangle(PROJECTOR_WIDTH + 24, PROJECTOR_HEIGHT / 2, 48, PROJECTOR_HEIGHT, {
      isStatic: true,
      label: "right-wall",
    });
    const ball = Matter.Bodies.circle(220, 100, 22, {
      restitution: 0.94,
      friction: 0.02,
      frictionAir: 0.002,
      label: "ball",
    });
    Matter.World.add(world, [floor, leftWall, rightWall, ball]);
    engineRef.current = engine;
    ballRef.current = ball;

    return () => {
      Matter.Engine.clear(engine);
      engineRef.current = null;
      ballRef.current = null;
    };
  }, []);

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

    const loop = (time: number) => {
      frameRef.current = requestAnimationFrame(loop);
      if (!running) {
        drawProjector(projectorCtx, engineRef.current, markers, homography, offset, calibrating, cameraPoints);
        return;
      }

      const delta = Math.min(16, time - lastTime);
      lastTime = time;
      drawCameraInput(cameraCtx, videoRef.current, demoMode, time);

      let currentMarkers = previousMarkersRef.current;
      if (cvReady && (cameraReady || demoMode)) {
        currentMarkers = demoMode
          ? animatedDemoMarkers(time)
          : detectMarkers(window.cv, cameraCanvas, hue, tolerance, saturation, value);
        currentMarkers = smoothMarkers(previousMarkersRef.current, currentMarkers);
        previousMarkersRef.current = currentMarkers;
      }

      updateColliderBodies(engineRef.current, currentMarkers, homography, offset);
      if (engineRef.current) {
        Matter.Engine.update(engineRef.current, delta);
        recycleBall(ballRef.current);
      }
      drawProjector(projectorCtx, engineRef.current, currentMarkers, homography, offset, calibrating, cameraPoints);
      drawCameraOverlay(cameraCtx, currentMarkers, cameraPoints, calibrating);

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
    calibrating,
    cvReady,
    demoMode,
    homography,
    hue,
    offset,
    running,
    saturation,
    tolerance,
    value,
  ]);

  const resetBall = () => {
    if (!ballRef.current) return;
    Matter.Body.setPosition(ballRef.current, { x: 220, y: 90 });
    Matter.Body.setVelocity(ballRef.current, { x: 9, y: -1 });
  };

  const handleCameraClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!calibrating) return;
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const point = {
      x: ((event.clientX - rect.left) / rect.width) * CAMERA_WIDTH,
      y: ((event.clientY - rect.top) / rect.height) * CAMERA_HEIGHT,
    };
    setCameraPoints((points) => {
      const next = points.length >= 4 ? [point] : [...points, point];
      if (next.length === 4) setCalibrating(false);
      return next;
    });
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
          <button type="button" onClick={resetBall}>
            <CirclePlus size={18} />
            ボール
          </button>
          <button
            type="button"
            onClick={() => {
              setCameraPoints([]);
              setCalibrating(true);
            }}
          >
            <Crosshair size={18} />
            4点補正
          </button>
          <button type="button" onClick={() => setCameraPoints(defaultCameraPoints)}>
            <RotateCcw size={18} />
            補正初期化
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
            <p>{calibrating ? "カメラ映像の四隅を順にクリック" : "検出した物体が静的コリジョンになります"}</p>
          </div>
          <SlidersHorizontal size={22} />
        </div>
        <div className="canvas-grid">
          <div className="canvas-block projector-block">
            <canvas ref={projectorCanvasRef} width={PROJECTOR_WIDTH} height={PROJECTOR_HEIGHT} />
          </div>
          <div className="canvas-block camera-block">
            <canvas
              ref={cameraCanvasRef}
              width={CAMERA_WIDTH}
              height={CAMERA_HEIGHT}
              onClick={handleCameraClick}
              className={calibrating ? "is-calibrating" : ""}
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
  calibrating: boolean,
) {
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
  calibrating: boolean,
  cameraPoints: Point[],
) {
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
      ctx.fillStyle = "#5aa7ff";
      ctx.beginPath();
      ctx.arc(body.position.x, body.position.y, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  });
}

function recycleBall(ball: Matter.Body | null) {
  if (!ball) return;
  if (ball.position.y > PROJECTOR_HEIGHT + 140 || ball.position.x < -140 || ball.position.x > PROJECTOR_WIDTH + 140) {
    Matter.Body.setPosition(ball, { x: 180 + Math.random() * 240, y: 40 });
    Matter.Body.setVelocity(ball, { x: 6 + Math.random() * 5, y: 0 });
  }
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
