"""Experimental Veritas blend: ConvNeXt primary + Benford + camera metadata."""

import gc
import re
from pathlib import Path

import numpy as np
import streamlit as st
import timm
import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import ExifTags, Image, ImageOps
from pillow_heif import register_heif_opener


st.set_page_config(page_title="Veritas Provenance Blend", layout="centered")

st.markdown(
    """
    <style>
      .block-container {
        max-width: 880px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }
      .veritas-hero {
        margin-bottom: 1.5rem;
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(37, 99, 235, 0.28);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(14, 165, 233, 0.04));
      }
      .veritas-eyebrow {
        margin: 0 0 0.35rem;
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
      }
      .veritas-hero h1 {
        margin: 0;
        font-size: clamp(2rem, 5vw, 2.7rem);
        line-height: 1.08;
      }
      .veritas-hero p:last-child {
        margin: 0.65rem 0 0;
        max-width: 650px;
        font-size: 1.04rem;
        opacity: 0.82;
      }
      [data-testid="stFileUploader"] {
        padding: 0.35rem 0.6rem;
        border: 1px dashed rgba(37, 99, 235, 0.42);
        border-radius: 14px;
        background: rgba(37, 99, 235, 0.035);
      }
      .stButton > button {
        min-height: 2.9rem;
        border: 0;
        border-radius: 10px;
        background: #2563eb;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
      }
      .stButton > button:hover {
        background: #1d4ed8;
        color: #ffffff;
      }
      [data-testid="stImage"] img {
        border-radius: 14px;
      }
      [data-testid="stExpander"] {
        border-radius: 12px;
      }
      .assessment-eyebrow {
        margin: 0 0 0.35rem;
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
      }
      @media (max-width: 640px) {
        .block-container {
          padding-top: 1rem;
          padding-left: 1rem;
          padding-right: 1rem;
        }
        .veritas-hero {
          padding: 1.1rem;
          border-radius: 14px;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "external_models" / "xrayon" / "AI Images Detector"
CHECKPOINT_PATH = EXTERNAL_ROOT / "checkpoints" / "checkpoint_phase2.pth"
MODEL_REPO_ID = "xRayon/convnext-ai-images-detector"
MODEL_FILENAME = "AI Images Detector/checkpoints/checkpoint_phase2.pth"
BENFORD_PROBS = np.array([np.log10(1 + 1 / digit) for digit in range(1, 10)])

AI_SOFTWARE_MARKERS = (
    "midjourney",
    "stable diffusion",
    "stablediffusion",
    "dall e",
    "dalle",
    "adobe firefly",
    "adobefirefly",
    "comfyui",
    "comfy ui",
)
HUMAN_LEANING_LABELS = {
    "HUMAN-LEANING",
    "LIKELY HUMAN",
    "STRONG HUMAN SIGNAL",
}

# Enables decoding of iPhone HEIC/HEIF uploads. It does not alter analysis.
register_heif_opener()


st.markdown(
    """
    <div class="veritas-hero">
      <p class="veritas-eyebrow">EXPLAINABLE IMAGE SCREENING</p>
      <h1>Veritas</h1>
      <p>Screen images for AI-generation signals and understand the evidence behind each assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def six_tier_label(signal):
  if signal >= 0.975:
    return "STRONG AI SIGNAL", "Strong AI-leaning evidence across the blended signal."
  if signal >= 0.965:
    return "LIKELY AI", "The blended evidence favors AI generation."
  if signal >= 0.90:
    return "AI-LEANING", "The visual signal is AI-leaning, but not conclusive."
  if signal >= 0.75:
    return "HUMAN-LEANING", "The evidence is mixed and slightly human-leaning."
  if signal >= 0.50:
    return "LIKELY HUMAN", "The blended evidence leans human."
  return "STRONG HUMAN SIGNAL", "The blended evidence shows a low AI signal."


def normalize_metadata_text(value):
  return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def has_metadata_value(metadata, key):
  value = normalize_metadata_text(metadata.get(key, ""))
  return value not in {"", "unknown", "none", "null", "n a", "na"}


def has_valid_camera_metadata(metadata):
  return (
      has_metadata_value(metadata, "Camera Make")
      and has_metadata_value(metadata, "Camera Model")
      and has_metadata_value(metadata, "Creation Timestamp")
  )


def detect_ai_software_tag(metadata):
  software = normalize_metadata_text(metadata.get("Software", ""))
  if not software:
    return False
  return any(marker in software for marker in AI_SOFTWARE_MARKERS)


def enforce_software_floor(label, explanation, ai_software_tag):
  if ai_software_tag and label in HUMAN_LEANING_LABELS:
    return (
        "AI-LEANING",
        "An AI-generator software tag was found, so the result cannot be classified as human-leaning.",
    )
  return label, explanation


def visual_evidence_summary(signal):
  """Human-readable display only; it does not change the detector score."""
  if signal >= 0.90:
    return "Strong AI-like signal"
  if signal >= 0.75:
    return "AI-like signal"
  if signal >= 0.50:
    return "Mixed signal"
  return "Low AI-like signal"


def benford_evidence_summary(points):
  """Human-readable display only; it does not change the Benford adjustment."""
  if points >= 0.02:
    return "More variation than expected"
  if points <= -0.02:
    return "Less variation than expected"
  return "Within the expected range"


def camera_evidence_summary(metadata):
  """Human-readable display only; it does not change the metadata adjustment."""
  if has_valid_camera_metadata(metadata):
    return "Camera details and capture time found"
  if (
      has_metadata_value(metadata, "Camera Make")
      and has_metadata_value(metadata, "Camera Model")
  ):
    return "Camera details found"
  if has_metadata_value(metadata, "Creation Timestamp"):
    return "Capture timestamp found"
  return "No camera information found"


def plain_language_takeaway(label):
  """Human-readable display only; it does not change the assessment."""
  if label in {"STRONG AI SIGNAL", "LIKELY AI", "AI-LEANING"}:
    return "What This Means: This image shows evidence consistent with AI generation."
  if label == "HUMAN-LEANING":
    return "What This Means: The evidence leans human, but some signals are mixed."
  return "What This Means: This image shows little evidence of AI generation."


@st.cache_resource
def load_detector():
  # Keep CPU inference and checkpoint loading within Streamlit Community
  # Cloud's memory budget without changing the detector or its output.
  torch.set_num_threads(1)
  checkpoint_path = CHECKPOINT_PATH
  if not checkpoint_path.exists():
    try:
      checkpoint_path = Path(
          hf_hub_download(
              repo_id=MODEL_REPO_ID,
              filename=MODEL_FILENAME,
              repo_type="model",
          )
      )
    except Exception as error:
      raise FileNotFoundError(
          "The local checkpoint is absent and the official xRayon checkpoint "
          "could not be downloaded automatically."
      ) from error
  model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
  checkpoint = torch.load(
      checkpoint_path,
      map_location="cpu",
      weights_only=True,
      mmap=True,
  )
  model.load_state_dict(checkpoint["model"])
  del checkpoint
  gc.collect()
  model.eval()
  transform = transforms.Compose([
      transforms.Resize(288),
      transforms.CenterCrop(256),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  return model, transform


def read_uploaded_image(uploaded_file):
  uploaded_file.seek(0)
  with Image.open(uploaded_file) as source:
    return ImageOps.exif_transpose(source).convert("RGB")


def extract_exif_metadata(uploaded_file):
  uploaded_file.seek(0)
  with Image.open(uploaded_file) as source:
    exif = source.getexif()

  wanted_tags = {
      "Camera Make": "Make",
      "Camera Model": "Model",
      "Software": "Software",
      "Creation Timestamp": "DateTimeOriginal",
      "Timestamp Timezone Offset": "OffsetTimeOriginal",
  }
  metadata = {}
  for label, tag_name in wanted_tags.items():
    tag_id = next(
        (key for key, name in ExifTags.TAGS.items() if name == tag_name),
        None,
    )
    value = exif.get(tag_id) if tag_id is not None else None
    if value not in (None, ""):
      metadata[label] = str(value)

  if "Creation Timestamp" not in metadata:
    for fallback in ("DateTimeDigitized", "DateTime"):
      tag_id = next(
          (key for key, name in ExifTags.TAGS.items() if name == fallback),
          None,
      )
      value = exif.get(tag_id) if tag_id is not None else None
      if value not in (None, ""):
        metadata["Creation Timestamp"] = str(value)
        break
  return metadata


def score_external_detector(image):
  model, transform = load_detector()
  pixels = transform(image).unsqueeze(0)
  with torch.inference_mode():
    probabilities = torch.softmax(model(pixels), dim=1)[0]
  return float(probabilities[1].item())


def leading_digit(value):
  text = str(abs(value)).replace(".", "").lstrip("0")
  if text and text[0] in "123456789":
    return int(text[0])
  return None


def benford_mad(image, grid_size=3):
  grayscale = np.asarray(image.convert("L"), dtype=float)
  height, width = grayscale.shape
  row_edges = np.linspace(0, height, grid_size + 1, dtype=int)
  column_edges = np.linspace(0, width, grid_size + 1, dtype=int)
  tile_scores = []

  for row in range(grid_size):
    for column in range(grid_size):
      tile = grayscale[
          row_edges[row] : row_edges[row + 1],
          column_edges[column] : column_edges[column + 1],
      ]
      differences = np.concatenate((
          np.diff(tile, axis=1).ravel(),
          np.diff(tile, axis=0).ravel(),
      ))
      digits = [
          digit
          for difference in differences
          if (digit := leading_digit(difference)) is not None
      ]
      if len(digits) < 50:
        tile_scores.append(0.0)
        continue
      counts = np.bincount(digits, minlength=10)[1:10]
      observed = counts / counts.sum() if counts.sum() else np.zeros(9)
      tile_scores.append(float(np.mean(np.abs(observed - BENFORD_PROBS))))

  return float(np.median(tile_scores)) if tile_scores else 0.0


def benford_adjustment(mad):
  """Keep Benford evidence deliberately small because it is not decisive alone."""
  normalized_drift = np.clip((mad - 0.045) / 0.045, -1.0, 1.0)
  return float(0.04 * normalized_drift)


def camera_metadata_adjustment(metadata):
  has_camera = (
      has_metadata_value(metadata, "Camera Make")
      and has_metadata_value(metadata, "Camera Model")
  )
  has_timestamp = has_metadata_value(metadata, "Creation Timestamp")
  if has_camera and has_timestamp:
    return 0.08, "Camera make/model and capture timestamp found"
  if has_camera:
    return 0.04, "Camera make/model found"
  if has_timestamp:
    return 0.02, "Camera capture timestamp found"
  return 0.0, "No camera metadata adjustment"


uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, or HEIC - up to 200 MB)",
    type=["png", "jpg", "jpeg", "heic", "heif"],
)
st.caption(
    "Best results: upload the original image file when possible. Screenshots "
    "of social-media pages, app interfaces, or images with large UI/text areas "
    "may be less reliable because Veritas analyzes the entire screenshot."
)

if uploaded_file is not None:
  image = read_uploaded_image(uploaded_file)
  st.image(image, use_container_width=True)

  if st.button("Analyze image", use_container_width=True):
    with st.spinner("Running visual, camera metadata, and Benford checks..."):
      try:
        external_signal = score_external_detector(image)
        metadata = extract_exif_metadata(uploaded_file)
        mad = benford_mad(image)
        benford_points = benford_adjustment(mad)
        camera_points, camera_note = camera_metadata_adjustment(metadata)
        ai_software_tag = detect_ai_software_tag(metadata)
        software_boost = 0.05 if ai_software_tag else 0.0

        final_signal = float(np.clip(
            external_signal
            + benford_points
            - camera_points
            + software_boost,
            0.0,
            1.0,
        ))

        label, explanation = six_tier_label(final_signal)
        label, explanation = enforce_software_floor(
            label,
            explanation,
            ai_software_tag,
        )

        with st.container(border=True):
          st.markdown('<p class="assessment-eyebrow">IMAGE AUTHENTICITY ASSESSMENT</p>', unsafe_allow_html=True)
          st.markdown(f"## {label}")
          st.write(explanation)
          st.caption(
              "Veritas provides an evidence-based screening result, not proof of an image's origin."
          )
          st.divider()
          st.markdown(
              "**What Influenced This Result:**\n\n"
              f"- **Visual Analysis:** {visual_evidence_summary(external_signal)}\n"
              f"- **Image Pattern Check:** {benford_evidence_summary(benford_points)}\n"
              f"- **Camera Information:** {camera_evidence_summary(metadata)}"
          )
          st.info(plain_language_takeaway(label))

          if ai_software_tag:
            st.warning(
                "AI-Generator Software Tag Detected. This metadata is editable and is treated as supporting evidence."
            )

        with st.expander("Why We Reached This Result", expanded=False):
          st.write(
              f"**Blended Evidence Score:** {final_signal:.3f} (not a probability)"
          )
          st.write(f"**Primary Visual Signal:** {external_signal * 100:.1f}%")
          st.write(f"**Pixel-Pattern Check:** {benford_points * 100:+.1f} points")
          st.write(f"**AI Software Metadata Adjustment:** {software_boost * 100:+.1f} points")
          st.write(
              f"**Camera Information Effect:** {-camera_points * 100:+.1f} points - {camera_note}"
          )
          st.caption(
              "Missing camera information is neutral. It can also be edited or copied and never proves an image is human-made."
          )

        with st.expander("Image Information", expanded=False):
          if metadata:
            for name, value in metadata.items():
              st.write(f"**{name}:** {value}")
            if (
                "Creation Timestamp" in metadata
                and "Timestamp Timezone Offset" not in metadata
            ):
              st.caption(
                  "Timezone: not embedded. The timestamp is camera-local time, not a verified location-based timezone."
              )
          else:
            st.info("No camera or creation metadata was found.")
      except Exception as error:
        st.error(f"Scan failed: {error}")
