# SignBridge

Real-time, privacy-first sign-language phrase recognition using computer vision and deep learning.

## Problem

People who use sign language may need to communicate with people who do not understand it in hospitals, public offices, colleges, shops, and emergency situations.

## Proposed solution

SignBridge is an assistive prototype that recognizes a defined vocabulary of sign-language phrases from a webcam and converts them into text. Optional text-to-speech output will make the result easier for another person to understand.

The system is designed to run locally so that camera data does not need to be uploaded to a remote server.

## Current scope

The first version will recognize a limited set of practical phrases. It is not intended to be a complete sign-language translator.

The project will investigate:

- Hand, pose, and facial landmark extraction
- Temporal sequence modeling
- Continuous webcam inference
- Confidence-aware predictions
- Prediction smoothing
- Person-level dataset splitting
- Local edge inference

## Planned pipeline

```text
Webcam
  ↓
Landmark extraction
  ↓
Frame sequence
  ↓
Temporal model
  ↓
Confidence filtering
  ↓
Phrase smoothing
  ↓
Text and optional speech
