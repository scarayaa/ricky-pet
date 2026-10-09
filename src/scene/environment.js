import * as THREE from "three";

const NOISE = /* glsl */ `
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
  }
  float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
    return v;
  }
`;

export const HORIZON = new THREE.Color("#ff2a9d");
const SKY_TOP = new THREE.Color("#07011a");
const SKY_MID = new THREE.Color("#2a0845");

/** Gradient sky dome with twinkling stars. */
export function createSky() {
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uTime: { value: 0 },
      uFade: { value: 0 },
      uTop: { value: SKY_TOP },
      uMid: { value: SKY_MID },
      uHorizon: { value: HORIZON },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime, uFade;
      uniform vec3 uTop, uMid, uHorizon;
      varying vec3 vDir;
      ${NOISE}
      void main() {
        float y = vDir.y;
        vec3 col = mix(uMid, uTop, smoothstep(0.0, 0.45, y));
        col = mix(col, uHorizon * 0.55, smoothstep(0.12, 0.0, y) * 0.8);

        // Stars on a lat/long grid, twinkling.
        vec2 sp = vec2(atan(vDir.z, vDir.x) * 120.0, y * 120.0);
        vec2 cell = floor(sp);
        float h = hash(cell);
        float star = step(0.985, h) * smoothstep(0.35, 0.0, length(fract(sp) - 0.5));
        star *= 0.6 + 0.4 * sin(uTime * (1.5 + h * 4.0) + h * 40.0);
        col += star * smoothstep(0.05, 0.3, y) * vec3(0.9, 0.85, 1.0);

        gl_FragColor = vec4(col * uFade, 1.0);
      }
    `,
  });
  return new THREE.Mesh(new THREE.SphereGeometry(400, 32, 16), material);
}

/** Striped synthwave sun with halo. */
export function createSun() {
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uRise: { value: 0 } },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv * 2.0 - 1.0;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime, uRise;
      varying vec2 vUv;
      void main() {
        // Sun sits in the inner 60% of the quad; the rest is halo.
        vec2 p = vUv / 0.6;
        p.y += (1.0 - uRise) * 1.6;
        float r = length(p);

        vec3 top = vec3(1.0, 0.78, 0.12);
        vec3 bot = vec3(1.0, 0.12, 0.55);
        vec3 col = mix(bot, top, smoothstep(-0.9, 0.8, p.y)) * 0.62;

        // Horizontal cuts that thicken toward the bottom and scroll down.
        float band = fract(p.y * 7.0 + uTime * 0.35);
        float gap = smoothstep(0.25, -0.9, p.y) * 0.75;
        float cuts = step(gap, band) + step(0.25, p.y);

        float disc = smoothstep(1.0, 0.985, r) * min(cuts, 1.0);
        float halo = exp(-max(r - 0.95, 0.0) * 3.2) * 0.18 * smoothstep(1.6, 0.9, r);
        vec3 c = col * disc + vec3(1.0, 0.2, 0.6) * halo;
        gl_FragColor = vec4(c * uRise, 1.0);
      }
    `,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
  mesh.scale.setScalar(80);
  mesh.position.set(0, 9, -170);
  return mesh;
}

/** Neon grid floor scrolling toward the camera, with wireframe-style mountains on the sides. */
export function createGrid() {
  const geometry = new THREE.PlaneGeometry(160, 160, 160, 160);
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, 0, -70);

  const material = new THREE.ShaderMaterial({
    extensions: { derivatives: true },
    uniforms: {
      uTime: { value: 0 },
      uDraw: { value: 0 },
      uSpeed: { value: 4 },
      uHorizon: { value: HORIZON },
    },
    vertexShader: /* glsl */ `
      uniform float uTime, uSpeed;
      varying vec2 vGrid;
      varying float vDepth, vHeight;
      ${NOISE}
      void main() {
        vec3 p = position;
        float scroll = uTime * uSpeed;
        vec2 g = vec2(p.x, p.z - scroll);
        float side = smoothstep(7.0, 30.0, abs(p.x));
        float h = fbm(g * 0.05) * 1.3 - 0.25;
        p.y += max(h, 0.0) * side * 22.0;
        vGrid = g;
        vHeight = p.y;
        vDepth = -p.z;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uDraw;
      uniform vec3 uHorizon;
      varying vec2 vGrid;
      varying float vDepth, vHeight;
      void main() {
        vec2 c = vGrid / 2.0;
        vec2 w = fwidth(c);
        vec2 g = abs(fract(c - 0.5) - 0.5) / (w * 1.2);
        float line = 1.0 - min(min(g.x, g.y), 1.0);

        // Lines draw in from the horizon toward the viewer.
        float reveal = smoothstep(150.0 * (1.0 - uDraw) - 8.0, 150.0 * (1.0 - uDraw), vDepth);
        float far = smoothstep(20.0, 140.0, vDepth);

        vec3 neon = mix(vec3(1.0, 0.16, 0.75), vec3(0.18, 0.88, 0.95), smoothstep(1.0, 14.0, vHeight));
        vec3 base = vec3(0.035, 0.0, 0.07);
        vec3 col = base + neon * line * 1.8 * reveal;
        col = mix(col, uHorizon * 0.45, far * 0.85);
        gl_FragColor = vec4(col * mix(0.15, 1.0, uDraw), 1.0);
      }
    `,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.y = -2.6;
  return mesh;
}
