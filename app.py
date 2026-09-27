"""
Mango Leaf Disease Detection - Gradio app for Hugging Face Spaces.

Run locally with:  python app.py
"""

import os
import glob
import gradio as gr

from prediction import (
    predict_disease,
    load_model,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DISEASE_INFO,
)

THRESHOLD_PCT = int(CONFIDENCE_THRESHOLD * 100)
MAX_PIXELS = 4000          # reject absurdly large images
MIN_PIXELS = 32            # reject tiny / corrupt images


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def format_uncertain(result):
    """Markdown shown when the model is below the confidence threshold."""
    rows = "\n".join(
        f"| {p['rank']} | {p['disease']} | {p['confidence']:.2f}% |"
        for p in result["top_predictions"]
    )
    return f"""
## ⚠️ Uncertain result

{result['uncertain_message']}

**Best guess:** {result['disease']} — only **{result['confidence']:.2f}%**, which is
below the {result['threshold']:.0f}% confidence threshold.

### Top {len(result['top_predictions'])} candidates

| Rank | Class | Confidence |
|---|---|---|
{rows}

Symptom and treatment details are withheld for low-confidence results, so that
no chemical is recommended on the strength of a weak guess.

**Tips for a better photo**
- One leaf, filling most of the frame
- Even, natural lighting with no harsh shadow
- Plain background (a sheet of paper works well)
- Camera steady and in focus
"""


def format_confident(result):
    """Markdown shown when the model is above the confidence threshold."""
    rows = "\n".join(
        f"| {p['rank']} | {p['disease']} | {p['confidence']:.2f}% |"
        for p in result["top_predictions"]
    )
    symptoms = "\n".join(f"- {s}" for s in result["symptoms"])
    treatment = "\n".join(f"- {m}" for m in result["medicine"])
    management = "\n".join(f"- {m}" for m in result["management"])

    pathogen = (
        f"{result['disease_type']} · *{result['pathogen']}*"
        if result["pathogen"] != "-"
        else result["disease_type"]
    )

    # When the green-leaf mask found nothing the percentage is meaningless,
    # so show that rather than a misleading 0%.
    if result.get("leaf_detected", True):
        infection = f"{result['infection_percentage']:.1f}%"
        severity_note = ""
    else:
        infection = "not estimated"
        severity_note = (
            "\n*Infected area could not be measured for this image — the leaf "
            "colour analysis found no green reference area, so severity is "
            "reported at its mildest band. Treat the severity figure with caution.*\n"
        )

    return f"""
## {result['disease']} — {result['confidence']:.2f}% confidence

{pathogen}

| Severity | Infection rate | Recovery chance |
|---|---|---|
| {result['severity']} | {infection} | {result['recovery']} |
{severity_note}
### Top {len(result['top_predictions'])} predictions

| Rank | Class | Confidence |
|---|---|---|
{rows}

A close second place means the model found those classes hard to tell apart.

### Typical symptoms
{symptoms}

### Recommended treatment
{treatment}

### Field management practices
{management}

---
*Preliminary screening only — not a final agricultural diagnosis. Confirm with a
local agricultural extension officer before applying any chemical.*
"""


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------
def analyze(image):
    """
    Analyse an uploaded leaf image.

    Returns (label_scores, markdown_report) for the two output components.
    """
    if image is None:
        return None, "Please upload a mango leaf image to begin."

    # Basic validation. Gradio already restricts the picker to images, but a
    # corrupt or extreme file can still arrive.
    width, height = image.size
    if width < MIN_PIXELS or height < MIN_PIXELS:
        return None, "That image is too small to analyse. Please upload a clearer photo."
    if width > MAX_PIXELS or height > MAX_PIXELS:
        return None, (
            f"That image is very large ({width}×{height}). "
            "Please resize it to under 4000 pixels on the longest side."
        )

    try:
        result = predict_disease(image)
    except FileNotFoundError as e:
        return None, f"**Model not available.** {e}"
    except Exception as e:
        return None, f"**Could not analyse that image.** {e}"

    report = (
        format_uncertain(result) if result["is_uncertain"] else format_confident(result)
    )
    return result["all_scores"], report


def get_examples():
    """One sample image per class, for the Examples gallery."""
    paths = sorted(glob.glob(os.path.join("samples", "*.jpg")))
    seen, chosen = set(), []
    for p in paths:
        # "Bacterial_Canker_1000.jpg" -> "Bacterial_Canker"
        key = os.path.basename(p).rsplit("_", 1)[0]
        if key not in seen:
            seen.add(key)
            chosen.append([p])
    return chosen


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
CSS = """
.disclaimer { font-size: 0.85rem; color: #777; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="Mango Leaf Disease Detection", css=CSS,
               theme=gr.themes.Soft(primary_hue="green")) as demo:

    gr.Markdown(
        """
        # 🥭 Mango Leaf Disease Detection

        Upload a mango leaf photo to identify likely diseases and get treatment
        guidance. The model recognises eight classes: Anthracnose, Bacterial
        Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew
        and Sooty Mould.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Mango leaf image",
                sources=["upload", "webcam", "clipboard"],
                height=320,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Analyze Disease", variant="primary")

            label_output = gr.Label(
                num_top_classes=3,
                label=f"Top 3 predictions (threshold {THRESHOLD_PCT}%)",
            )

        with gr.Column(scale=1):
            report_output = gr.Markdown(
                value="Upload an image and press **Analyze Disease** to see results."
            )

    examples = get_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            outputs=[label_output, report_output],
            fn=analyze,
            cache_examples=False,
            label="Sample leaf images — click one to try it",
        )

    with gr.Accordion("About this project", open=False):
        gr.Markdown(
            f"""
            **How it works**

            1. The uploaded image is resized to 300×300 and normalised.
            2. A CNN classifier (EfficientNetB0, TensorFlow/Keras) predicts a
               probability for each of the eight classes.
            3. Infected leaf area is estimated separately with HSV colour
               thresholding in OpenCV, giving a severity band.
            4. Treatment advice is looked up from the predicted class and severity.

            **Confidence threshold**

            If the top prediction scores below **{THRESHOLD_PCT}%**, the app reports the
            result as uncertain and withholds treatment advice, rather than
            presenting a weak guess as a finding.

            Note that this model is **poorly calibrated**: on clear leaf images it
            usually reports close to 100%, so the threshold rarely triggers in
            practice and mainly catches badly degraded inputs. The percentage
            should be read as "which class won", not as a true probability of
            being correct. Calibrating the model (for example with temperature
            scaling on a validation set) would make this figure meaningful.

            **Limitations**

            - Trained only on mango leaves; other plants give meaningless output.
            - The model cannot say "this is not a leaf" — it always assigns one
              of the eight classes, which is why the threshold matters.
            - Severity comes from colour thresholding, so lighting and background
              affect it. On a fully browned leaf no green reference area is found
              and the infection rate is reported as "not estimated".
            - This is a preliminary screening tool for a course project, not a
              substitute for expert agricultural diagnosis.
            """
        )

    clear_btn.add([image_input, label_output, report_output])

    submit_btn.click(fn=analyze, inputs=image_input,
                     outputs=[label_output, report_output])
    image_input.change(fn=analyze, inputs=image_input,
                       outputs=[label_output, report_output])


if __name__ == "__main__":
    # Warm the model at start-up so the first user request is not slow.
    try:
        load_model()
        print(f"Model loaded. Classes: {CLASS_NAMES}")
    except Exception as e:
        print(f"Warning: model not pre-loaded ({e})")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
