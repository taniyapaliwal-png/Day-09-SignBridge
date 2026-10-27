<!--
  SIGNBRIDGE // CYBERPUNK LAB
  GitHub README edition
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:080B18,45:382078,100:00D9FF&height=180&section=header&text=SignBridge&fontSize=56&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=YOUR%20HANDS%20HAVE%20A%20VOICE&descAlignY=60&descSize=16" width="100%" alt="SignBridge — Your hands have a voice"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=15&duration=2800&pause=800&color=62E7FF&center=true&vCenter=true&width=600&lines=Computer+Vision+%2F%2F+Accessibility;Detect.+Recognize.+Communicate.;Built+locally.+Designed+for+impact." alt="Animated SignBridge tagline"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks-00BFA6?style=for-the-badge)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![Tests](https://img.shields.io/badge/Tests-5%20passing-00C896?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

<br/>

**A privacy-first, real-time sign recognition prototype.**

Turning webcam hand landmarks into readable predictions — one sign at a time.

<br/>

[ ✦ Explore the project](#-how-it-works) · [⚡ Quick start](#-quick-start) · [🧪 Testing](#-testing) · [🗺️ Roadmap](#-roadmap)

</div>

---

## `01` // SYSTEM STATUS

<table>
<tr>
<td width="33%" align="center">

### <code>02</code>

**Trained labels**

`HELLO` · `THANK_YOU`

</td>
<td width="33%" align="center">

### <code>05/05</code>

**Automated tests**

All currently passing

</td>
<td width="33%" align="center">

### <code>LOCAL</code>

**Processing**

Webcam inference on-device

</td>
</tr>
</table>

> [!NOTE]
> SignBridge is an experimental prototype, not a complete sign-language translator. The current classifier recognizes two trained labels; its accuracy and generalization to unseen signers still need systematic evaluation.

## `02` // THE MISSION

Communication shouldn't feel like a locked door.

People who use sign language can face communication barriers when interacting with people who don't know the same signs. SignBridge explores how computer vision can help bridge part of that gap.

**The goal:** build a locally processed, accessible tool that recognizes a defined vocabulary and displays its predicted meaning as text.

No cloud inference is required by the current pipeline. Your webcam frames are processed locally.

## `03` // HOW IT WORKS

<div align="center">

```text
   ┌──────────────────┐
   │   WEBCAM INPUT   │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ MEDIAPIPE HANDS  │
   │  21 LANDMARKS    │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ NUMERICAL FEATURES│
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ TRAINED CLASSIFIER│
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ CONFIDENCE +     │
   │ PREDICTION SMOOTHING│
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │  ON-SCREEN TEXT  │
   └──────────────────┘
```

</div>

<details>
<summary><b>⌘ What happens behind the scenes?</b></summary>

1. OpenCV reads webcam frames.
2. MediaPipe detects hands and extracts their landmarks.
3. The landmarks are converted into the feature columns expected by the trained model.
4. The classifier predicts a label.
5. When probability estimates are supported, the application displays model confidence and filters low-confidence predictions.
6. A rolling window of recent predictions reduces rapid label changes.
7. The resulting label appears in the live webcam window.

The current system uses hand landmarks. Temporal sequence models, full-body pose recognition, and facial-expression analysis are future possibilities, not existing features.

</details>

## `04` // FEATURE MODULES

<table>
<tr>
<td width="50%">

### ◉ Live vision

Real-time webcam input and hand tracking with OpenCV and MediaPipe.

</td>
<td width="50%">

### ⌁ Landmark intelligence

Converts detected hand landmarks into numerical model features.

</td>
</tr>
<tr>
<td>

### ⟳ Prediction smoothing

Uses recent predictions to reduce flickering labels.

</td>
<td>

### ◈ Confidence feedback

Shows model probability estimates when the classifier supports them.

</td>
</tr>
<tr>
<td>

### ▦ Dataset validation

Checks CSV structure, labels, hand flags, numeric values, and finite coordinates.

</td>
<td>

### ⛨ Local-first design

Processes webcam frames locally and keeps the dataset and trained model out of Git tracking.

</td>
</tr>
</table>

## `05` // TECH STACK

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFA6?style=flat-square)](https://ai.google.dev/edge/mediapipe)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)

</div>

## `06` // QUICK START

<details open>
<summary><b>⚡ Installation — macOS</b></summary>

**Requirements:** Python 3.11 recommended, Git, and a webcam.

**1. Clone the repository**

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SignBridge
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

**4. Allow camera access**

If macOS prompts you, allow webcam access for your terminal or IDE.

</details>

## `07` // RUN THE PIPELINE

<details>
<summary><b>01 — Collect landmark samples</b></summary>

```bash
python3 src/data/collect_landmarks.py HELLO
python3 src/data/collect_landmarks.py THANK_YOU
```

Follow the collector's on-screen instructions to save samples. The resulting CSV is intended to stay on your local machine.

</details>

<details>
<summary><b>02 — Validate the dataset</b></summary>

```bash
python3 src/data/validate_dataset.py
```

The current local dataset contains 57 structurally valid samples across two labels. Dataset validation does not verify that every sample is correctly signed or labeled.

</details>

<details>
<summary><b>03 — Train the classifier</b></summary>

```bash
python3 models/train_model.py
```

This creates the local trained model used by live prediction.

</details>

<details>
<summary><b>04 — Start live recognition</b></summary>

```bash
python3 src/vision/live_prediction.py
```

Hold a collected sign in view and watch the predicted label. Press `Q` to exit.

</details>

<details>
<summary><b>05 — Run automated tests</b></summary>

```bash
python3 -m pytest tests/ -v
```

The current dataset-validator test suite has **5 passing tests**.

</details>

## `08` // REPOSITORY MAP

```text
SignBridge/
├── .github/
│   └── workflows/
├── data/
│   └── README.md
├── models/
│   └── train_model.py
├── src/
│   ├── data/
│   │   ├── collect_landmarks.py
│   │   └── validate_dataset.py
│   └── vision/
│       ├── webcam_landmarks.py
│       └── live_prediction.py
├── tests/
│   └── test_validate_dataset.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The landmark CSV and generated `.joblib` model are local artifacts and are not intended to be committed.

## `09` // ROADMAP

- [x] Webcam hand landmark extraction
- [x] Numerical landmark collection
- [x] Dataset validation
- [x] Two-label classifier training
- [x] Live prediction with smoothing
- [x] Automated validator tests
- [ ] Evaluate on a separate test set
- [ ] Improve sample diversity and test unseen signers
- [ ] Explore temporal modeling for dynamic signs
- [ ] Expand the vocabulary using a defined sign-language standard
- [ ] Explore optional local text-to-speech
- [ ] Improve the user interface and accessibility

## `10` // PRIVACY & RESPONSIBLE USE

- Webcam frames are processed locally by the current application.
- Landmark datasets and trained model artifacts are excluded from Git tracking.
- Check `.gitignore` and Git history before publishing.
- Model confidence is not proof of correctness.
- SignBridge is experimental and must not be relied upon for emergency or other high-stakes communication.

This project is an exploration of assistive technology, not a replacement for qualified sign-language interpreters.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D9FF,50:382078,100:080B18&height=100&section=footer" width="100%" alt=""/>

### `SIGNBRIDGE // END OF TRANSMISSION`

**Small signs. Big connections.**

Built with curiosity, computer vision, and accessibility in mind. ♡

</div>
