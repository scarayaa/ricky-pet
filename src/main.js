import * as THREE from "three";
import { createSky, createSun, createGrid } from "./scene/environment.js";
import { createFace } from "./scene/face.js";
import { createComposer } from "./scene/post.js";

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const canvas = document.getElementById("scene");
const bootEl = document.getElementById("boot");
const titleEl = document.getElementById("title");
const hintEl = document.getElementById("hint");

// --- Renderer -------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.toneMapping = THREE.NoToneMapping;
const maxDpr = Math.min(window.devicePixelRatio, 2);
let dpr = Math.min(maxDpr, 1.5);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
camera.position.set(0, 0.4, 7);

const sky = createSky();
const sun = createSun();
const grid = createGrid();
scene.add(sky, sun, grid);

const post = createComposer(renderer, scene, camera);

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setPixelRatio(dpr);
  renderer.setSize(w, h, false);
  post.setSize(w, h, dpr);
  camera.aspect = w / h;
  // Keep the face framed on portrait screens by pulling the camera back.
  camera.position.z = w / h < 0.8 ? 7 / Math.max(w / h / 0.8, 0.55) : 7;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// --- Face -----------------------------------------------------------------
const pivot = new THREE.Group();
const PIVOT_Y = 0.75;
pivot.position.set(0, PIVOT_Y, -0.6);
pivot.scale.setScalar(0.8);
scene.add(pivot);
let face = null;
const faceReady = createFace({ floorY: (grid.position.y - PIVOT_Y) / 0.8 - 0.2 }).then((mesh) => {
  face = mesh;
  pivot.add(mesh);
  return mesh;
});

// --- Boot terminal ----------------------------------------------------------
const bootLines = [
  "> BOOT SEQUENCE // RICKY.EXE v1.0",
  "> INITIALIZING NEON GRID ............ OK",
  "> LOCATING SUBJECT ................. RICCARDO",
  "> RECONSTRUCTING HEAD MESH ..........",
  "> 33153 VERTICES // 65536 POLYGONS",
  "> DIGITAL ENTITY ONLINE_",
];
async function typeBoot() {
  if (reducedMotion) {
    bootEl.textContent = bootLines.join("\n");
    return;
  }
  for (const line of bootLines) {
    for (let i = 0; i <= line.length; i++) {
      bootEl.textContent = bootEl.textContent.replace(/[^\n]*$/, line.slice(0, i));
      await new Promise((r) => setTimeout(r, 14));
    }
    bootEl.textContent += "\n";
    await new Promise((r) => setTimeout(r, 160));
  }
}

// --- Interaction ------------------------------------------------------------
const pointer = new THREE.Vector2();
// ?yaw=<radians> pins a base rotation, handy to inspect the sides and back.
const baseYaw = Number(new URLSearchParams(location.search).get("yaw")) || 0;
const drag = { active: false, x: 0, yaw: 0, velocity: 0 };
canvas.addEventListener("pointermove", (e) => {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  if (drag.active) {
    const dx = (e.clientX - drag.x) / innerWidth;
    drag.yaw += dx * 4;
    drag.velocity = dx * 4;
    drag.x = e.clientX;
  }
});
canvas.addEventListener("pointerdown", (e) => {
  drag.active = true;
  drag.x = e.clientX;
  canvas.setPointerCapture(e.pointerId);
});
// Double tap/click detection that also works on touch screens.
let lastTap = 0;
canvas.addEventListener("pointerup", (e) => {
  drag.active = false;
  if (e.timeStamp - lastTap < 300) replay();
  lastTap = e.timeStamp;
});

// --- Timeline ---------------------------------------------------------------
const ease = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
let introStart = performance.now();
let glitch = 0;
let nextGlitch = 9;
let titleShown = false;

function replay() {
  introStart = performance.now() - 1800; // skip the environment fade on replays
  glitch = 1;
  titleShown = false;
  titleEl.classList.remove("show");
}

function triggerTitleGlitch() {
  const h1 = titleEl.querySelector("h1");
  h1.classList.add("glitch");
  setTimeout(() => h1.classList.remove("glitch"), 220);
}

// --- Adaptive resolution ---------------------------------------------------
let frames = 0;
let fpsWindow = performance.now();
function adaptDpr(now) {
  frames++;
  if (now - fpsWindow < 1500) return;
  const fps = (frames * 1000) / (now - fpsWindow);
  frames = 0;
  fpsWindow = now;
  const next = fps < 48 ? Math.max(0.75, dpr - 0.25) : fps > 58 ? Math.min(maxDpr, dpr + 0.25) : dpr;
  if (next !== dpr) {
    dpr = next;
    resize();
  }
}

// --- Loop -------------------------------------------------------------------
const clock = new THREE.Clock();
const speed = reducedMotion ? 6 : 1;

function frame(now) {
  const dt = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;
  const t = ((now - introStart) / 1000) * speed;

  sky.material.uniforms.uTime.value = time;
  sky.material.uniforms.uFade.value = ease(t / 1.5);
  sun.material.uniforms.uTime.value = time;
  sun.material.uniforms.uRise.value = ease((t - 0.6) / 2.6);
  grid.material.uniforms.uTime.value = time;
  grid.material.uniforms.uDraw.value = ease((t - 0.2) / 2.2);

  if (face) {
    const u = face.material.uniforms;
    u.uTime.value = time;
    u.uAssemble.value = Math.min(Math.max((t - 2.0) / 3.6, 0), 1);

    if (!titleShown && t > 5.4) {
      titleShown = true;
      titleEl.classList.add("show");
      hintEl.classList.add("show");
      bootEl.classList.add("done");
      triggerTitleGlitch();
    }

    // Random glitch bursts once assembled.
    if (!reducedMotion && time > nextGlitch && t > 6) {
      glitch = 0.8 + Math.random() * 0.4;
      nextGlitch = time + 6 + Math.random() * 8;
      triggerTitleGlitch();
    }
    glitch = Math.max(0, glitch - dt * 2.5);
    u.uGlitch.value = glitch;

    // Idle float, pointer look-at, drag spin with inertia.
    if (!drag.active) {
      drag.velocity *= 0.94;
      drag.yaw += drag.velocity;
      drag.yaw *= 0.985; // ease back to facing forward
    }
    const targetY = baseYaw + pointer.x * 0.45 + drag.yaw;
    const targetX = -pointer.y * 0.2;
    pivot.rotation.y += (targetY - pivot.rotation.y) * 0.08;
    pivot.rotation.x += (targetX - pivot.rotation.x) * 0.08;
    pivot.position.y = PIVOT_Y + Math.sin(time * 1.1) * 0.06;
  }

  camera.position.x += (pointer.x * 0.35 - camera.position.x) * 0.04;
  camera.lookAt(0, -0.1, 0);

  post.retro.uniforms.uTime.value = time;
  post.retro.uniforms.uGlitch.value = glitch + (t < 2.2 ? (1 - t / 2.2) * 0.6 : 0);

  post.composer.render(dt);
  adaptDpr(now);
}

Promise.all([faceReady, document.fonts?.ready]).finally(() => {
  introStart = performance.now();
  clock.start();
  typeBoot();
  renderer.setAnimationLoop(frame);
});
