# ricky-pet

## Scopo
Web app 3D retrowave: all'apertura appaiono griglia neon, sole a strisce e montagne wireframe, e la testa 3D di Riccardo si assembla da frammenti di griglia digitale. È un regalo/gioco per un amico.

## Stack
- Three.js (r170) + Vite 6, JavaScript vanilla, shader GLSL custom.
- Pipeline della testa in Python: MediaPipe FaceLandmarker + segmentazione multiclasse (capelli/pelle) + SDF della testa, numpy/scipy.
- Versioni fissate in `mise.toml` (node 22, python 3.12, venv in `.venv`).

## Comandi
```bash
mise install                          # runtime alle versioni del progetto
pip install mediapipe pillow numpy scipy  # dentro il venv di mise (solo per rigenerare la testa)
npm install
tools/fetch_models.sh                 # modelli MediaPipe (non versionati)
npm run head                          # rigenera public/face/* da reference/front.jpg (+ three-quarter.jpg per il colore dei capelli)
npm run head -- --photo <path>        # usa un'altra foto frontale
npm run dev                           # dev server
npm run build                         # build in dist/
```

## Convenzioni
- Codice e commenti in inglese, UI in italiano.
- Foto: `assets/photos/all/riccardo-NN.jpg` (originali rinominati, mappa in `manifest.csv`), pose scelte in `assets/photos/reference/`.
- Gli asset generati vanno in `public/face/` e si committano (la build web non richiede Python).
- `assets/photos/` è in `.gitignore`: le foto personali non vanno mai nel repo (pubblico). Il deploy è su GitHub Pages via `.github/workflows/deploy.yml` a ogni push su `main`; `vite.config.js` usa `base: "./"`, quindi i percorsi degli asset devono essere relativi (`import.meta.env.BASE_URL`).

## Architettura
- `tools/build_head.py`: foto frontale → landmark 3D + segmentazione → testa SDF (cranio, mascella, collo, orecchie) campionata come r(θ,φ) su griglia deformata (più densa sul viso) → viso reale fuso nel cranio → volume dei capelli → texture: foto proiettata dove è visibile, capelli/pelle/barba sintetizzati altrove. Output `public/face/face.bin` (pos/uv/normal/index), `face.json` (+ chinY, minY/maxY), `face.jpg`.
- `tools/models/`: modelli MediaPipe (face_landmarker.task, selfie_multiclass.tflite) e topologia canonica del viso.
- `src/scene/environment.js`: cielo, sole e griglia/montagne (tutto procedurale, uno shader per oggetto).
- `src/scene/face.js`: carica la mesh della testa (origine = centro del cranio), la rende non indicizzata per animare per triangolo; assemblaggio, ologramma, glitch e collo che sfuma in griglia negli shader.
- `src/scene/post.js`: bloom a metà risoluzione + pass retro (aberrazione cromatica, glitch, grana) + OutputPass. Le scanline sono in CSS.
- `src/main.js`: timeline dell'intro, interazione (puntatore, trascinamento, doppio tocco), DPR adattivo in base agli FPS. `?yaw=<rad>` fissa una rotazione per ispezionare lati e nuca.

## Decisioni prese
- 2026-10-09 — Ricostruzione da una sola foto frontale con MediaPipe invece della fotogrammetria — le foto sono per lo più di gruppo con pose diverse, la fotogrammetria non convergerebbe.
- 2026-10-09 — Mesh pre-calcolata offline in binario — nessun ML nel browser, caricamento ~1,6 MB (~1,2 MB gzip) per 33k vertici.
- 2026-10-09 — Testa completa ricostruita in locale (SDF + viso fuso + texture sintetizzata) invece di servizi image-to-3D esterni — niente foto di Riccardo inviate a terzi, nessun account/API key.
- 2026-10-09 — Repo pubblico (GitHub Pages gratuito) senza foto né modelli MediaPipe; Riccardo ha dato il consenso alla pubblicazione del volto.
- 2026-10-09 — Skill Three.js da CloudAI-X/threejs-skills (MIT) in `.claude/skills/`.
