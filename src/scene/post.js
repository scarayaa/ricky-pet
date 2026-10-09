import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

/** Chromatic aberration + glitch tearing + grain, in one fullscreen pass. */
const RetroShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uGlitch: { value: 0 },
    uAberration: { value: 0.0025 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uTime, uGlitch, uAberration;
    varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec2 uv = vUv;
      float row = floor(uv.y * 48.0);
      float tear = step(1.0 - uGlitch * 0.35, hash(vec2(row, floor(uTime * 24.0))));
      uv.x += tear * (hash(vec2(row, uTime)) - 0.5) * 0.08 * uGlitch;

      vec2 dir = uv - 0.5;
      float amt = uAberration * (1.0 + uGlitch * 6.0);
      vec3 col;
      col.r = texture2D(tDiffuse, uv + dir * amt).r;
      col.g = texture2D(tDiffuse, uv).g;
      col.b = texture2D(tDiffuse, uv - dir * amt).b;

      col += (hash(uv * 900.0 + uTime) - 0.5) * 0.035;
      gl_FragColor = vec4(col, 1.0);
    }
  `,
};

export function createComposer(renderer, scene, camera) {
  const size = renderer.getSize(new THREE.Vector2());
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  // Bloom at half resolution: the blur hides it and it halves the fill cost.
  const bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), 0.7, 0.5, 0.7);
  composer.addPass(bloom);

  const retro = new ShaderPass(RetroShader);
  composer.addPass(retro);
  composer.addPass(new OutputPass());

  return {
    composer,
    retro,
    setSize(w, h, dpr) {
      composer.setPixelRatio(dpr);
      composer.setSize(w, h);
      bloom.resolution.set(w / 2, h / 2);
    },
  };
}
