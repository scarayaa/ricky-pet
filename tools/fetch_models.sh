#!/usr/bin/env bash
# Download the MediaPipe models used by tools/build_head.py.
set -euo pipefail
cd "$(dirname "$0")/models"
curl -sSfL -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -sSfL -o selfie_multiclass.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
echo "Models ready in tools/models/"
