
# SignBridge

**A privacy-first, real-time sign recognition prototype built with Python and computer vision.**

SignBridge uses webcam input and hand landmarks to recognize a small, predefined vocabulary of signs and display predictions on screen. The project is being developed as a foundation for accessible, locally processed communication tools.

> **Current status:** Working prototype with two trained labels: `HELLO` and `THANK_YOU`.

## Problem

Communication can be difficult when people who use sign language interact with people who do not understand it. SignBridge explores how computer vision can help bridge a small part of that communication gap.

## Current Features

- **Live webcam inference** using OpenCV and MediaPipe.
- **Hand landmark extraction** using MediaPipe Hands.
- **Landmark-based classification** using a trained machine-learning model.
- **Prediction smoothing** to reduce rapid changes in displayed labels.
- **Confidence feedback** when the trained model supports probability estimates.
- **Dataset validation** to check CSV structure, labels, numeric values, and hand-presence flags.
- **Local processing:** webcam frames are processed locally and are not intentionally uploaded to a remote server.

## How It Works

```text
Webcam
   ↓
MediaPipe hand landmarks
   ↓
Numerical landmark features
   ↓
Trained classifier
   ↓
Confidence feedback
   ↓
Smoothed sign prediction
   ↓
On-screen text
```

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn (classifier, depending on the training script)
- Joblib
- pytest

## Project Structure

```text
SignBridge/
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
└── README.md
```

## Setup

Python 3.11 is the version currently used during development.

1. Clone the repository and enter the project folder:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd SignBridge
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m pip install pytest
   ```

4. Allow webcam access when prompted by macOS.

## Run the Project

### Collect landmark samples

```bash
python3 src/data/collect_landmarks.py HELLO
python3 src/data/collect_landmarks.py THANK_YOU
```

Follow the on-screen instructions to capture samples. The collector stores numerical landmarks in `data/landmarks.csv`.

### Validate the dataset

```bash
python3 src/data/validate_dataset.py
```

### Train the model

```bash
python3 models/train_model.py
```

### Run live prediction

```bash
python3 src/vision/live_prediction.py
```

Press **Q** to exit the webcam window.

### Run tests

```bash
python3 -m pytest tests/ -v
```

## Data Privacy

- Webcam frames are processed locally by the application.
- The landmark dataset and trained model are intended to remain local and are excluded from Git tracking.
- Do not commit personal landmark datasets, model artifacts, videos, or images.
- Review `.gitignore` and Git status before pushing changes.

## Limitations

- The current prototype recognizes only two trained labels.
- Predictions depend on the quality and variety of the collected samples.
- Model confidence is not a guarantee that a prediction is correct.
- This is not a complete sign-language translator. It does not currently understand continuous signing, grammar, or the full use of facial expressions and body posture.
- The current pipeline uses hand landmarks; temporal sequence modeling and text-to-speech remain future work unless implemented separately.

## Roadmap

- [ ] Improve dataset diversity and evaluate generalization to unseen signers.
- [ ] Add a reliable train/test evaluation pipeline.
- [ ] Explore temporal modeling for dynamic signs and phrases.
- [ ] Improve prediction feedback and user interface.
- [ ] Explore optional local text-to-speech output.
- [ ] Expand the vocabulary using a clearly defined sign-language standard.

## Responsible Use

SignBridge is an experimental accessibility project, not a substitute for a qualified interpreter. Its predictions should not be relied upon for emergencies or other high-stakes communication.

## License

See [LICENSE](LICENSE).
```

## Before we commit

One important check: this README assumes the project paths and commands match your current files. Save it, then run:

```bash
python3 -m pytest tests/ -v
git status --short
```

If the tests still pass, tell me. We'll verify that the README doesn't promise anything your code doesn't do, then commit and push the final version.