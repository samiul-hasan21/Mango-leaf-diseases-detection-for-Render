"""
Mango Leaf Disease Detection - prediction logic.

Works on in-memory images (PIL / NumPy) so it can be driven by Gradio
without writing temporary files to disk.
"""

import os
import numpy as np
import cv2
from PIL import Image

# TFLite interpreter only -- NOT full TensorFlow. Importing full TensorFlow
# costs ~620 MB by itself before a model is even loaded, which alone exceeds
# Render's free-tier 512 MB limit and crashes the process. The TFLite runtime
# does the same inference for ~35 MB, an ~18x reduction.
from ai_edge_litert.interpreter import Interpreter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = (300, 300)
CONFIDENCE_THRESHOLD = 0.60      # below this the result is reported as uncertain
TOP_K = 3

UNCERTAIN_MESSAGE = (
    "The model is not confident about this image. Please retake the photo "
    "(single leaf, good lighting, plain background) or consult an agricultural "
    "expert before acting on this result."
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mango_model.tflite")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "model", "class_names.txt")

DEFAULT_CLASS_NAMES = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould",
]

# ---------------------------------------------------------------------------
# Disease knowledge base
# ---------------------------------------------------------------------------
DISEASE_INFO = {
    "Anthracnose": {
        "type": "Fungal disease",
        "pathogen": "Colletotrichum gloeosporioides",
        "symptoms": [
            "Small dark brown to black spots that enlarge and merge",
            "Sunken lesions on leaves, often with a lighter centre",
            "Leaf tips and margins dry out and turn dark",
            "Premature leaf drop; blossom blight in flowering season",
        ],
        "management": [
            "Prune and burn infected leaves, twigs and fallen debris",
            "Avoid overhead irrigation; keep the canopy dry",
            "Improve spacing and pruning for better air circulation",
            "Apply a protectant fungicide spray before the wet season",
        ],
    },
    "Bacterial Canker": {
        "type": "Bacterial disease",
        "pathogen": "Xanthomonas campestris pv. mangiferaeindicae",
        "symptoms": [
            "Water-soaked spots that turn dark and angular",
            "Raised cankers with gum or bacterial ooze on stems",
            "Lesions often bounded by leaf veins",
            "Yellow halo around older spots; leaves may fall early",
        ],
        "management": [
            "Prune infected branches well below the canker and burn them",
            "Disinfect pruning tools between cuts",
            "Avoid injuring the tree; wounds are the main entry point",
            "Avoid overhead irrigation and working in the orchard when wet",
        ],
    },
    "Cutting Weevil": {
        "type": "Insect pest damage",
        "pathogen": "Deporaus marginatus (mango leaf-cutting weevil)",
        "symptoms": [
            "Leaves cut across in a characteristic straight or curved line",
            "Chewed edges and irregular holes in young leaves",
            "New flush shoots damaged most heavily",
            "Cut leaf pieces found on the ground under the tree",
        ],
        "management": [
            "Collect and destroy fallen cut leaves to break the life cycle",
            "Hand-pick adult weevils during early flush stages",
            "Monitor new flush closely, as that is when attack peaks",
            "Use an approved insecticide only if damage is widespread",
        ],
    },
    "Die Back": {
        "type": "Fungal disease",
        "pathogen": "Lasiodiplodia theobromae (Botryodiplodia)",
        "symptoms": [
            "Twigs and branches dry backwards from the tip",
            "Leaves turn brown, curl and stay attached to dead twigs",
            "Dark discolouration of wood visible when a twig is cut",
            "Gum exudation from affected branches",
        ],
        "management": [
            "Prune affected branches 15-20 cm below the visibly dead tissue",
            "Seal large cut surfaces with a protective paste",
            "Remove and burn all pruned material",
            "Correct water stress and poor drainage, which worsen dieback",
        ],
    },
    "Gall Midge": {
        "type": "Insect pest damage",
        "pathogen": "Procontarinia matteiana and related species",
        "symptoms": [
            "Small raised wart-like galls scattered over the leaf surface",
            "Galls turn brown or black and the tissue around them dies",
            "Leaves become distorted, brittle and may drop early",
            "Exit holes visible on galls after larvae emerge",
        ],
        "management": [
            "Collect and destroy heavily galled leaves before larvae emerge",
            "Plough or rake soil under the tree to expose pupae",
            "Monitor during the new flush period when egg-laying occurs",
            "Use an approved insecticide at flush stage if infestation is high",
        ],
    },
    "Healthy": {
        "type": "No disease detected",
        "pathogen": "-",
        "symptoms": [
            "Uniform green colour with no spots or lesions",
            "Intact leaf margins with no cutting or chewing damage",
            "Smooth surface with no galls, powder or sooty coating",
            "Normal leaf shape without curling or distortion",
        ],
        "management": [
            "Continue regular watering and balanced fertilisation",
            "Inspect new flush periodically for early signs of pests",
            "Maintain orchard sanitation and adequate spacing",
            "Keep monitoring; healthy today does not mean immune",
        ],
    },
    "Powdery Mildew": {
        "type": "Fungal disease",
        "pathogen": "Oidium mangiferae",
        "symptoms": [
            "White to greyish powdery coating on the leaf surface",
            "Coating appears first on young leaves and flower panicles",
            "Affected tissue turns purplish-brown underneath the powder",
            "Curling and premature drop of young leaves and flowers",
        ],
        "management": [
            "Remove and destroy severely affected leaves and panicles",
            "Improve air movement through pruning; avoid dense canopies",
            "Avoid excess nitrogen, which encourages soft susceptible growth",
            "Apply a sulphur-based or approved fungicide at flowering",
        ],
    },
    "Sooty Mould": {
        "type": "Secondary fungal growth",
        "pathogen": "Capnodium spp. growing on insect honeydew",
        "symptoms": [
            "Black powdery or crust-like coating on the upper leaf surface",
            "Coating can be wiped or washed off, unlike a true leaf spot",
            "Sticky honeydew present, often with ants moving on the leaves",
            "Reduced photosynthesis causing weak, pale growth",
        ],
        "management": [
            "Control the underlying sap-sucking insects (aphids, mealybugs, scale)",
            "Wash affected leaves with water or a mild soap solution",
            "Manage ants, which protect the honeydew-producing insects",
            "Prune to open the canopy and improve light penetration",
        ],
    },
}

RECOMMENDATIONS = {
    "Anthracnose": {
        "Mild": {"recovery": "Very High", "medicine": ["Copper Fungicide", "Neem Oil"]},
        "Moderate": {"recovery": "High", "medicine": ["Mancozeb", "Copper Oxychloride"]},
        "Severe": {"recovery": "Medium", "medicine": ["Propiconazole", "Remove infected leaves", "Destroy infected plant debris"]},
    },
    "Bacterial Canker": {
        "Mild": {"recovery": "High", "medicine": ["Copper Hydroxide Spray", "Neem Oil"]},
        "Moderate": {"recovery": "Medium", "medicine": ["Streptocycline", "Copper Oxychloride"]},
        "Severe": {"recovery": "Low", "medicine": ["Prune infected branches", "Apply Bordeaux Mixture", "Avoid overhead irrigation"]},
    },
    "Cutting Weevil": {
        "Mild": {"recovery": "Very High", "medicine": ["Neem Oil Spray", "Manual insect removal"]},
        "Moderate": {"recovery": "High", "medicine": ["Imidacloprid", "Lambda-Cyhalothrin"]},
        "Severe": {"recovery": "Medium", "medicine": ["Systemic Insecticide", "Destroy infected leaves", "Field sanitation"]},
    },
    "Die Back": {
        "Mild": {"recovery": "High", "medicine": ["Copper Oxychloride", "Prune affected twigs"]},
        "Moderate": {"recovery": "Medium", "medicine": ["Carbendazim", "Thiophanate Methyl"]},
        "Severe": {"recovery": "Low", "medicine": ["Remove severely infected branches", "Apply systemic fungicide", "Improve drainage"]},
    },
    "Gall Midge": {
        "Mild": {"recovery": "Very High", "medicine": ["Neem Extract Spray", "Yellow sticky traps"]},
        "Moderate": {"recovery": "High", "medicine": ["Dimethoate", "Malathion"]},
        "Severe": {"recovery": "Medium", "medicine": ["Systemic insecticide", "Destroy affected leaves", "Regular field monitoring"]},
    },
    "Healthy": {
        "Healthy": {"recovery": "Healthy Plant", "medicine": ["No treatment needed", "Maintain regular watering", "Use balanced fertilizer"]},
    },
    "Powdery Mildew": {
        "Mild": {"recovery": "Very High", "medicine": ["Sulfur Spray", "Neem Oil"]},
        "Moderate": {"recovery": "High", "medicine": ["Myclobutanil", "Wettable Sulfur"]},
        "Severe": {"recovery": "Medium", "medicine": ["Systemic Fungicide", "Remove infected leaves", "Improve air circulation"]},
    },
    "Sooty Mould": {
        "Mild": {"recovery": "Very High", "medicine": ["Wash leaves with water", "Neem Oil Spray"]},
        "Moderate": {"recovery": "High", "medicine": ["Copper Fungicide", "Control aphids and mealybugs"]},
        "Severe": {"recovery": "Medium", "medicine": ["Systemic insecticide", "Prune heavily affected leaves", "Use horticultural oil"]},
    },
}


def load_class_names():
    """Load class names from model/class_names.txt, falling back to defaults."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()

_interpreter = None
_input_details = None
_output_details = None


def load_model():
    """Load (and cache) the TFLite interpreter."""
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure model/mango_model.tflite is committed to the repo."
            )
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()[0]
        _output_details = _interpreter.get_output_details()[0]
    return _interpreter


def to_pil(image):
    """Accept a PIL image, NumPy array or file path and return RGB PIL."""
    if image is None:
        return None
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    return image.convert("RGB")


def get_severity(percent):
    """Map an infection percentage onto a severity band."""
    if percent == 0:
        return "Healthy"
    if percent <= 20:
        return "Mild"
    if percent <= 50:
        return "Moderate"
    return "Severe"


def analyze_infection(pil_image):
    """
    Estimate infected leaf area using HSV colour thresholds.

    Returns (percentage, leaf_detected). When the green-leaf mask finds nothing
    -- a fully browned leaf, a dark photo, or an unusual background -- the
    percentage is meaningless, so the caller is told detection failed rather
    than being handed a misleading 0%.
    """
    arr = np.array(pil_image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, IMG_SIZE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    disease_mask = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([35, 255, 255]))

    leaf_pixels = int(np.sum(leaf_mask > 0))
    disease_pixels = int(np.sum(disease_mask > 0))

    if leaf_pixels == 0:
        return 0.0, False
    return round(min((disease_pixels / leaf_pixels) * 100, 100), 2), True


def get_top_k_predictions(scores, k=TOP_K):
    """Return the k highest-scoring classes, ranked descending."""
    k = min(k, len(scores))
    top_indices = np.argsort(scores)[-k:][::-1]
    return [
        {
            "rank": pos + 1,
            "disease": CLASS_NAMES[int(idx)],
            "confidence": round(float(scores[idx]) * 100, 2),
        }
        for pos, idx in enumerate(top_indices)
    ]


def predict_disease(image):
    """
    Run a full analysis on an image.

    image : PIL.Image, NumPy array, or path to an image file
    returns: dict with prediction, top-K, severity and advice
    """
    pil = to_pil(image)
    if pil is None:
        raise ValueError("No image provided.")

    load_model()  # ensures the interpreter is loaded and cached

    resized = pil.resize(IMG_SIZE)
    arr = np.asarray(resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    _interpreter.set_tensor(_input_details["index"], arr)
    _interpreter.invoke()
    scores = _interpreter.get_tensor(_output_details["index"])[0]

    top_predictions = get_top_k_predictions(scores, TOP_K)
    predicted_index = int(np.argmax(scores))
    confidence = float(np.max(scores))
    disease = CLASS_NAMES[predicted_index]
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    infection_percentage, leaf_detected = analyze_infection(pil)
    severity = get_severity(infection_percentage)

    # A diseased class can never sit in the "Healthy" severity band. That
    # happens when the green-leaf mask finds nothing (fully browned leaf, dark
    # photo), which previously produced an empty treatment list. Fall back to
    # the mildest real band so advice is always returned.
    if disease != "Healthy" and severity == "Healthy":
        severity = "Mild"

    if disease == "Healthy":
        rec = RECOMMENDATIONS["Healthy"]["Healthy"]
    else:
        disease_recs = RECOMMENDATIONS.get(disease, {})
        rec = disease_recs.get(severity) or disease_recs.get("Mild") or {
            "recovery": "Unknown",
            "medicine": [],
        }

    info = DISEASE_INFO.get(
        disease, {"type": "Unknown", "pathogen": "-", "symptoms": [], "management": []}
    )

    return {
        "disease": disease,
        "confidence": round(confidence * 100, 2),
        "infection_percentage": infection_percentage,
        "leaf_detected": leaf_detected,
        "severity": severity,
        "recovery": rec["recovery"],
        "medicine": rec["medicine"],
        "top_predictions": top_predictions,
        # label -> probability, the format gr.Label expects
        "all_scores": {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))},
        "is_uncertain": is_uncertain,
        "threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
        "uncertain_message": UNCERTAIN_MESSAGE if is_uncertain else "",
        "disease_type": info["type"],
        "pathogen": info["pathogen"],
        "symptoms": info["symptoms"],
        "management": info["management"],
    }
