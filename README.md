<div align="center">

# A Neuro-Symbolic Agent for Video Anomaly Detection

**Alessia Donata Camarda · Giovambattista Ianni · Angelo Laface · Simona Perri**

Submitted to
<strong>C.A.R.L.A: Workshop on Cognitive Architectures for Robotics: LLMs and Logic in Action</strong><br>
July 18, 2026 · Lisbon, Portugal

</div>

---

> **Abstract.**  
The field of video anomaly detection has evolved towards more flexible approaches that increasingly leverage the semantic understanding capabilities of Vision-Language Models (VLMs). These models are particularly appealing as they enable reasoning about the high-level semantics of anomalous events, going beyond simple pattern deviations. However, despite their strong semantic capabilities, VLMs are often computationally expensive and may be unsuitable for deployment in high-risk domains. In this paper, we propose a neuro-symbolic agent that combines a YOLO-based perception model for object detection with a symbolic reasoning component capable of identifying anomalies formalized through explicit rules. The reasoning module also serves as a filtering mechanism, selecting ambiguous situations that require deeper semantic analysis according to user-defined criteria. Such cases are then delegated to a VLM, reducing unnecessary reliance on these costly models. Finally, we discuss the limitations of the proposed approach and illustrate its applicability through a representative real-world scenario.

---

## Overview

This repository contains the implementation and resources related to the paper
**“A Neuro-Symbolic Agent for Video Anomaly Detection”**.

The proposed system combines neural perception and symbolic reasoning for video anomaly detection. 
A YOLO-based perception module extracts relevant objects from video frames and represents them as symbolic facts. 
These facts are processed by a **DP-SR reasoning module**, which relies on **Answer Set Programming** under the hood. 
By means of user-defined rules, the system can detect explicitly modeled anomalies and select visually ambiguous situations that require further semantic interpretation by a Vision-Language Model.

> [!NOTE]
> The current implementation should be considered a proof of concept. Its purpose is to 
> demonstrate the feasibility of the proposed neuro-symbolic approach and to illustrate 
> how neural perception, symbolic reasoning, and Vision-Language Models can be combined 
> for video anomaly detection. It is not intended as a final or production-ready version 
> of the system.

## Architecture

The pipeline is composed of the following main components:

1. **Perception Module**  
   Detects objects in video frames using YOLO and extracts symbolic facts such as
   object identity, class, position, and semantic area membership.

2. **Reasoning Module**  
   Uses user-defined ASP-based stream reasoning rules, expressed in DP-SR, to detect explicitly modeled anomalies and candidate anomalies over time.

3. **Vision-Language Model Module**  
   Receives ambiguous candidate anomalies and performs higher-level semantic
   interpretation.

> [!NOTE] 
> The use of a Vision-Language Model is motivated by the fact that some events can be
> detected or selected through symbolic rules, but cannot always be conclusively
> classified as anomalous using symbolic information alone. For example, an object
> detector may recognize the presence of a knife in a kitchen, but it cannot determine
> whether the knife is being used normally for cooking or in a potentially dangerous
> way. In such cases, the symbolic reasoning module can identify the situation as a
> candidate anomaly, while the Vision-Language Model provides a higher-level
> visual-semantic interpretation of the scene.

## Requirements

- `uv`
- Linux x86-64 and Java 11 or newer on `PATH` for DP-SR
- [Git LFS](https://git-lfs.github.com) — required to download large binary files


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DeMaCS-UNICAL/NS_VAD-CARLA2026.git
cd NS_VAD-CARLA2026
```

### 2. Install dependencies

```bash
uv sync --extra cuda
# or
uv sync --extra cpu
```

Choose the appropriate extra based on your hardware:

- `cuda` — recommended when a compatible NVIDIA GPU is available
- `cpu` — use if CUDA is not available, though performance may be significantly lower

### 3. Run the agent

```bash
GOOGLE_API_KEY=... uv run python agent.py --scenario docs/examples/wrong_way
```

By default, the agent uses the YOLO weights at `./models/yolo26x.pt` to detect the objects present in the scene. If the file is not present, Ultralytics downloads the weights automatically on first use. You can provide a different weights file with `--yolo-model`.

You can use one of the predefined scenarios in the `docs/examples/` folder, or define your own.

---

## Scenarios

A scenario defines the runtime configuration of the agent. Each scenario is a folder containing:

- `rules.lp` — logic rules for anomaly detection
- `areas.json` *(optional)* — spatial areas of interest used to interpret object positions
- A video file (e.g. `video.mp4`) — the video stream to analyze

The agent starts DP-SR automatically, connects to it over local loopback sockets, and writes the output file `outputs/agent.log`

---

## Vision-Language Model

The agent currently supports **Google AI Studio** as the Vision-Language Model (VLM) provider, using `gemini-2.5-flash` by default. You can override this with `--vlm-model`.

To enable VLM-based analysis, generate an API key from [Google AI Studio](https://aistudio.google.com/) and provide it at runtime via `--vlm-api-key` or the `GOOGLE_API_KEY` environment variable.

### Input modes

By default, each `candidate_anomaly` interval is sent to the VLM as a video clip. To send sampled frames instead, use: `--vlm-input-mode frames`

To inspect the frames or video clips sent to the Vision-Language Model, use: `--save-vlm-input`

## Resources

- [**Predicates**](docs/predicates.md): documentation of the supported predicates used to identify anomalies.
- [**Tools**](tools/): utilities for defining semantic areas in video frames and visualizing detected anomalies.
- [**Examples**](docs/examples/): a few example encodings and videos.
- [**DP-SR documentation**](http://ares.mat.unical.it:20140/en/language): documentation for the DP-SR stream reasoning system.
