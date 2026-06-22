---
title: BenchHub Leaderboards
emoji: 🏆
colorFrom: purple
colorTo: indigo
sdk: static
app_file: index.html
pinned: true
hf_oauth: false
short_description: Live ML leaderboards — vision, audio, NLP. Submit free.
tags:
  - leaderboard
  - benchmark
  - evaluation
  - model-evaluation
  - computer-vision
  - object-detection
  - image-segmentation
  - depth-estimation
  - stereo
  - audio-classification
  - automatic-speech-recognition
  - speech-recognition
  - question-answering
  - named-entity-recognition
  - token-classification
datasets:
  - scene_parse_150
  - ashraq/esc50
  - openslr/librispeech_asr
  - rafaelpadilla/coco2017
---

# 🏆 BenchHub Leaderboards

A read-only mirror of **public** leaderboard standings from
**[runbenchhub.com](https://runbenchhub.com)** — an open, multi-modal model
benchmarking platform. Browse the standings here; **run your own model and
submit on BenchHub** (free).

## What's benchmarked
Live boards across a growing set of domains, each with real models scored on
the same eval set:

- **Vision** — Image Classification, Semantic Segmentation (mIoU), Object
  Detection (mAP), Monocular & **Stereo** Depth Estimation, Image Captioning
- **Audio** — Audio Classification (ESC-50), Automatic Speech Recognition (WER)
- **NLP** — Text Classification, Extractive Question Answering (SQuAD),
  Named-Entity Recognition (CoNLL-2003), Visual Question Answering

## Submit your model
Every **"Submit"** button here deep-links to that leaderboard's submission page
on BenchHub. Submitting is free — sign in with **GitHub, Google, or 🤗 Hugging
Face**, run the one-line client on your predictions, and see where you rank.

> This Space holds **no** ground-truth samples, predictions, or submission UI —
> it reads a derived standings dataset (`HF_RESULTS_REPO`) that BenchHub
> publishes. The full interactive experience (per-sample explorer, GT
> visualizations, model comparison) lives on
> **[runbenchhub.com](https://runbenchhub.com)**.
