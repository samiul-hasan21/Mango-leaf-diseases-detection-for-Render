---
title: Mango Leaf Disease Detection
emoji: 🥭
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Detect 8 mango leaf diseases from a photo using deep learning
---

# 🥭 Mango Leaf Disease Detection

A web application that identifies diseases in mango leaves from a photograph,
built with TensorFlow/Keras and Gradio.

## Classes

The model recognises eight categories:

| Class | Type |
|---|---|
| Anthracnose | Fungal disease |
| Bacterial Canker | Bacterial disease |
| Cutting Weevil | Insect pest damage |
| Die Back | Fungal disease |
| Gall Midge | Insect pest damage |
| Healthy | No disease |
| Powdery Mildew | Fungal disease |
| Sooty Mould | Secondary fungal growth |

## How it works

1. **Preprocessing** — the uploaded image is resized to 300×300 and converted to a tensor.
2. **Classification** — an EfficientNetB0-based CNN predicts a probability for each class.
3. **Top-3 predictions** — the three highest-scoring classes are shown with confidence bars,
   so a close second place is visible rather than hidden.
4. **Confidence threshold** — if the top score is below 60%, the result is reported as
   *uncertain* and treatment advice is withheld, so no chemical is recommended on a weak guess.
5. **Severity estimation** — infected leaf area is estimated with HSV colour thresholding
   in OpenCV and mapped to Mild / Moderate / Severe.
6. **Guidance** — symptoms, treatment and field management practices are looked up for the
   predicted class and severity.

## Deploying to Render (free, no account-age or paid-plan restrictions)

1. Push this folder to a GitHub repo (add your `model/mango_model.keras` first —
   it's 21 MB, well under GitHub's 100 MB limit, so no Git LFS needed).
2. Go to https://render.com and sign up (GitHub login works).
3. Click **New +** → **Blueprint**, and select your repo. Render will detect
   `render.yaml` automatically and configure the service for you.
   (Alternatively: **New +** → **Web Service**, connect the repo, set build
   command `pip install -r requirements.txt` and start command `python app.py`.)
4. Click **Create Web Service**. First build takes 5-10 minutes.
5. Once live, Render gives you a URL like `https://mango-leaf-disease-detection.onrender.com`.

**Free tier notes:**
- The service sleeps after 15 minutes of inactivity; the next visit takes
  ~30-60 seconds to wake up.
- Free plan has 512 MB RAM. TensorFlow plus this model should fit, but if the
  build crashes with an out-of-memory error, that's the free tier's limit —
  either upgrade the plan or convert the model to TensorFlow Lite to shrink
  its memory footprint.

## Running locally

```bash
git clone <your-repo-url>
cd mango-leaf-disease-detection
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860

## Project structure

```
.
├── app.py                    # Gradio interface
├── prediction.py             # Model loading, inference, severity, knowledge base
├── model/
│   ├── mango_model.keras     # Trained model (Git LFS)
│   └── class_names.txt       # Class label order
├── samples/                  # Example leaf images
├── requirements.txt
├── .gitattributes            # Git LFS tracking for model weights
└── README.md
```

## Limitations

- Trained only on mango leaves. Images of other plants, or of non-plants, still
  return one of the eight classes — the model has no "none of the above" option.
- The trained model is **highly overconfident**: on in-distribution leaf images it
  usually reports close to 100%, so the 60% threshold rarely triggers in practice.
  It mainly catches badly degraded inputs. Calibration (e.g. temperature scaling)
  would make the confidence figure more meaningful.
- Severity comes from colour thresholding, so lighting, shadow and background
  strongly affect the infection percentage.
- This is a preliminary screening tool built as a course project. It is not a
  substitute for diagnosis by a qualified agricultural expert, and treatment
  suggestions are generic rather than dosage-specific.

## Model training

Trained with TensorFlow/Keras using an EfficientNetB0 backbone on a mango leaf
disease dataset, with GPU acceleration on Kaggle.

## License

MIT — for educational use.
