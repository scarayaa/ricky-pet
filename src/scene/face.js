import * as THREE from "three";

/**
 * Loads the baked head mesh (tools/build_head.py) and turns it into a holographic
 * object whose triangles assemble from grid-snapped shards rising off the floor.
 * The origin is the skull center, so rotations turn the head on its neck.
 */
export async function createFace({ base = `${import.meta.env.BASE_URL}face/`, floorY = -2.8 } = {}) {
  const [meta, bin, texture] = await Promise.all([
    fetch(base + "face.json").then((r) => r.json()),
    fetch(base + "face.bin").then((r) => r.arrayBuffer()),
    new THREE.TextureLoader().loadAsync(base + "face.jpg"),
  ]);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;

  const n = meta.vertexCount;
  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bin, meta.position, n * 3), 3));
  indexed.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(bin, meta.uv, n * 2), 2));
  indexed.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(bin, meta.normal, n * 3), 3));
  indexed.setIndex(new THREE.BufferAttribute(new Uint16Array(bin, meta.index, meta.indexCount), 1));

  // Per-triangle attributes need unshared vertices.
  const geometry = indexed.toNonIndexed();
  indexed.dispose();
  const pos = geometry.attributes.position.array;
  const count = pos.length / 3;
  const centroid = new Float32Array(count * 3);
  const rand = new Float32Array(count);
  for (let t = 0; t < count; t += 3) {
    const r = Math.random();
    const cx = (pos[t * 3] + pos[t * 3 + 3] + pos[t * 3 + 6]) / 3;
    const cy = (pos[t * 3 + 1] + pos[t * 3 + 4] + pos[t * 3 + 7]) / 3;
    const cz = (pos[t * 3 + 2] + pos[t * 3 + 5] + pos[t * 3 + 8]) / 3;
    for (let k = 0; k < 3; k++) {
      centroid[(t + k) * 3] = cx;
      centroid[(t + k) * 3 + 1] = cy;
      centroid[(t + k) * 3 + 2] = cz;
      rand[t + k] = r;
    }
  }
  geometry.setAttribute("aCentroid", new THREE.BufferAttribute(centroid, 3));
  geometry.setAttribute("aRand", new THREE.BufferAttribute(rand, 1));
  geometry.computeBoundingSphere();

  const material = new THREE.ShaderMaterial({
    extensions: { derivatives: true },
    uniforms: {
      uMap: { value: texture },
      uTime: { value: 0 },
      uAssemble: { value: 0 },
      uGlitch: { value: 0 },
      uFloorY: { value: floorY },
      uMinY: { value: meta.minY },
      uMaxY: { value: meta.maxY },
      // The neck dissolves into the hologram below this height.
      uNeckFade: { value: meta.chinY - 0.55 },
    },
    vertexShader: /* glsl */ `
      uniform float uTime, uAssemble, uGlitch, uFloorY, uMinY, uMaxY;
      attribute vec3 aCentroid;
      attribute float aRand;
      varying vec2 vUv;
      varying vec3 vNormal, vLocal, vView;
      varying float vT;

      float hash(float n) { return fract(sin(n) * 43758.5453); }

      void main() {
        // Bottom-up wave with per-shard jitter; reaches 1 everywhere at uAssemble = 1.
        float yN = clamp((aCentroid.y - uMinY) / (uMaxY - uMinY), 0.0, 1.0);
        float delay = yN * 0.55 + aRand * 0.3;
        float t = smoothstep(delay, delay + 0.3, uAssemble * 1.15);
        float e = 1.0 - pow(1.0 - t, 3.0);

        // Start: centroid snapped to a voxel grid, scattered and dropped onto the floor.
        vec3 scatter = vec3(hash(aRand * 91.0) - 0.5, 0.0, hash(aRand * 37.0) - 0.5) * vec3(7.0, 0.0, 5.0);
        vec3 start = floor((aCentroid + scatter) * 4.0) / 4.0;
        start.y = uFloorY + hash(aRand * 13.0) * 0.4;

        vec3 c = mix(start, aCentroid, e);
        // Shards rise in an arc and spin into place.
        c.y += sin(e * 3.14159) * (0.8 + aRand);
        float s = mix(0.35, 1.0, e);
        vec3 p = c + (position - aCentroid) * s;

        // Glitch: horizontal slice displacement.
        float slice = floor(p.y * 9.0 + uTime * 30.0);
        p.x += uGlitch * (hash(slice) - 0.5) * 0.5 * step(0.7, hash(slice + 1.0));

        vec4 world = modelMatrix * vec4(p, 1.0);
        vUv = uv;
        vNormal = normalize(mat3(modelMatrix) * normal);
        vLocal = position;
        vView = normalize(cameraPosition - world.xyz);
        vT = t;
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uMap;
      uniform float uTime, uAssemble, uGlitch, uMinY, uMaxY, uNeckFade;
      varying vec2 vUv;
      varying vec3 vNormal, vLocal, vView;
      varying float vT;

      float gridLine(vec2 c) {
        vec2 w = fwidth(c);
        vec2 g = abs(fract(c - 0.5) - 0.5) / (w * 1.1);
        return 1.0 - min(min(g.x, g.y), 1.0);
      }

      void main() {
        if (vT < 0.002) discard;
        vec3 cyan = vec3(0.18, 0.9, 0.95);
        vec3 pink = vec3(1.0, 0.16, 0.62);

        vec3 tex = texture2D(uMap, vUv).rgb;
        // Retro grade: lift contrast, push shadows magenta and highlights warm.
        float lum = dot(tex, vec3(0.299, 0.587, 0.114));
        tex = mix(tex, tex * tex * 1.6, 0.25);
        tex += pink * 0.035 * (1.0 - lum) + vec3(0.06, 0.02, 0.0) * lum;

        vec3 n = normalize(vNormal);
        float fres = pow(1.0 - max(dot(n, vView), 0.0), 2.5);
        vec3 rim = mix(pink, cyan, smoothstep(-0.4, 0.6, n.x)) * fres * 0.8;

        // Holographic grid in face space + travelling scan band.
        float grid = gridLine(vLocal.xy * 9.0) + gridLine(vec2(vLocal.z * 9.0, vLocal.y * 9.0)) * 0.5;
        float span = uMaxY - uMinY + 0.6;
        float scanY = uMaxY + 0.3 - mod(uTime * 0.45, span);
        float band = exp(-pow((vLocal.y - scanY) * 10.0, 2.0));

        float solid = smoothstep(0.75, 1.0, vT);
        // Before turning solid, only the lattice lines are drawn.
        float shardGrid = gridLine(vLocal.xy * 5.0);
        if (solid < 0.01 && shardGrid < 0.35 && vT < 0.97) discard;
        vec3 holo = cyan * (0.6 + shardGrid * 1.2) + pink * fres;
        vec3 col = mix(holo, tex + rim, solid);
        col += cyan * grid * (0.05 + band * 0.6) * solid;
        col += cyan * band * 0.12 * solid;
        col += pink * uGlitch * grid;

        // Neck: solid fades out into grid lines, which thin out toward the bottom.
        float neck = smoothstep(uNeckFade - 0.35, uNeckFade, vLocal.y);
        if (neck < 1.0) {
          float lines = shardGrid * smoothstep(uNeckFade - 1.4, uNeckFade - 0.2, vLocal.y);
          if (neck < 0.02 && lines < 0.35) discard;
          col = mix(cyan * lines * 1.3, col, neck);
        }

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false; // shards start outside the final bounds
  return mesh;
}
