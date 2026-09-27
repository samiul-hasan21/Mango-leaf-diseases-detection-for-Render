# 🥭 Mango Leaf Disease Detection

A web application that identifies diseases in mango leaves from a photograph, built with TensorFlow/Keras for classification and Gradio for the interface. Deployed live on Render.

**🔗 Live demo: [https://mango-leaf-disease-detection.onrender.com/](https://mango-leaf-disease-detection.onrender.com/)**

> Note: the app runs on Render's free tier and sleeps after 15 minutes of inactivity. The first request after a period of inactivity may take 30–60 seconds to wake it up — this is normal, not a bug.

## What it does

Upload a photo of a mango leaf, and the app:

1. Classifies it into one of eight categories
2. Shows the **top 3 predictions** with confidence scores, not just the top guess
3. Withholds treatment advice and flags the result as **uncertain** if confidence is below 60%, rather than presenting a weak guess as a finding
4. Estimates infection severity from leaf discoloration
5. Provides symptoms, treatment options, and field management practices for the predicted class

## Classes

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
2. **Classification** — a CNN classifier (TensorFlow/Keras) predicts a probability for each of the eight classes.
3. **Confidence threshold** — if the top score is below 60%, the result is reported as uncertain and treatment advice is withheld.
4. **Severity estimation** — infected leaf area is estimated with HSV colour thresholding in OpenCV and mapped to Mild / Moderate / Severe.
5. **Guidance** — symptoms, treatment, and field management practices are looked up for the predicted class and severity.

## Tech stack

- **Model**: TensorFlow / Keras (CNN image classifier)
- **Interface**: Gradio
- **Image analysis**: OpenCV (HSV-based severity estimation)
- **Hosting**: Render (free tier, deployed via `render.yaml`)

## Project structure

```
.
├── app.py                    # Gradio interface
├── prediction.py              # Model loading, inference, severity, knowledge base
├── model/
│   ├── mango_model.keras     # Trained model
│   └── class_names.txt        # Class label order
├── samples/                    # Example leaf images for the demo gallery
├── requirements.txt
├── render.yaml                  # Render deployment config
└── README.md
```

## Running locally

```bash
git clone https://github.com/samiul-hasan21/Mango-leaf-diseases-detection-for-Render.git
cd Mango-leaf-diseases-detection-for-Render
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860

## Deploying your own copy

This repo deploys to [Render](https://render.com) via the included `render.yaml`:

1. Fork or clone this repo
2. Sign up at render.com (GitHub login works)
3. Click **New +** → **Blueprint**, select this repo
4. Render reads `render.yaml` automatically and deploys on the free tier

## Limitations

- Trained only on mango leaves — images of other plants still return one of the eight classes, since the model has no "none of the above" option.
- The model is not well calibrated: on clear leaf images it typically reports close to 100% confidence, so the 60% threshold mainly catches badly degraded inputs rather than genuinely ambiguous cases.
- Severity comes from colour thresholding, so lighting and background affect the estimate. On a fully browned leaf, no green reference area is found and the infection rate is reported as "not estimated."
- This is a preliminary screening tool built as a course project — not a substitute for diagnosis by a qualified agricultural expert, and treatment suggestions are generic rather than dosage-specific.

## License

MIT — for educational use.
