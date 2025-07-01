# mel_spectrogram_viewer.py
"""
Interactive Streamlit application to visualize mel spectrograms and play their
corresponding audio for ESC-50 and ICBHI datasets.

This application allows users to select samples in three ways:
1. By specific indices (e.g., '5 10 15').
2. By count (e.g., '50' to see the first 50 samples).
3. By range (e.g., '50,150' to see samples 50 through 149).

The application will display all requested samples in a grid. A warning is
shown for large requests (>50 samples).

Run the application from a terminal:
    streamlit run mel_spectrogram_viewer.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Dataset Configuration (Extensible Part)
# -------------------------------------------------------------------------
class DatasetPaths(TypedDict):
    spec_dir: str
    audio_dir: str

DATASET_CONFIG: Dict[str, DatasetPaths] = {
    "esc-50": {
        "spec_dir": "esc-50/entire_spec_npy_8000",
        "audio_dir": "esc-50/audio",
    },
    "icbhi": {
        "spec_dir": "icbhi/entire_spec_npy_8000",
        "audio_dir": "icbhi/ICBHI_final_database",
    },
}

# -------------------------------------------------------------------------
# Data Handling
# -------------------------------------------------------------------------
class DatasetLoader:
    """
    Lazy loader for pre-computed mel spectrograms and their associated audio.
    """
    def __init__(self, dataset_name: str, root_dir: str = "datasets") -> None:
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from {sorted(DATASET_CONFIG.keys())}."
            )
        config = DATASET_CONFIG[self.dataset_name]
        base_path = Path(root_dir)
        self.spec_dir = base_path / config["spec_dir"]
        self.audio_dir = base_path / config["audio_dir"]

        if not self.spec_dir.is_dir():
            raise FileNotFoundError(f"Spectrogram directory not found: '{self.spec_dir}'")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: '{self.audio_dir}'")

        self._file_paths: List[Path] = sorted(self.spec_dir.glob("*.npy"))
        if not self._file_paths:
            raise RuntimeError(f"No spectrogram (.npy) files found in '{self.spec_dir}'.")

    def __len__(self) -> int:
        return len(self._file_paths)

    def _find_audio_path(self, spec_path: Path) -> Path | None:
        spec_stem = spec_path.stem
        for ext in [".wav", ".flac", ".mp3"]:
            audio_path = self.audio_dir / (spec_stem + ext)
            if audio_path.exists():
                return audio_path
        return None

    def load(self, idx: int) -> Tuple[np.ndarray, str, Path | None]:
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} is out of bounds. Valid range is 0 to {len(self) - 1}.")
        spec_path = self._file_paths[idx]
        spec_array = np.load(spec_path)
        audio_path = self._find_audio_path(spec_path)
        return spec_array, spec_path.name, audio_path

# -------------------------------------------------------------------------
# Streamlit UI Helpers
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data loader...")
def get_loader(name: str) -> DatasetLoader:
    return DatasetLoader(name)

def parse_specific_indices(text: str) -> List[int]:
    """Convert a string like "0, 3 9" into a list of unique ints."""
    if not text:
        return []
    indices: set[int] = set()
    raw_parts = text.replace(",", " ").split()
    for part in raw_parts:
        try:
            indices.add(int(part))
        except ValueError:
            continue
    return sorted(list(indices))

def parse_range_or_count(text: str) -> List[int]:
    """Parses 'N' as range(N) and 'N,M' as range(N, M)."""
    text = text.strip()
    if not text:
        return []

    parts = [p.strip() for p in text.split(',') if p.strip()]
    try:
        if len(parts) == 1:
            count = int(parts[0])
            return list(range(count))
        elif len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if start >= end:
                st.warning("Start of range must be less than end. Swapping values.")
                start, end = end, start
            return list(range(start, end))
        else:
            st.warning("Invalid format for Range/Count. Use 'N' or 'N, M'.")
            return []
    except ValueError:
        st.warning(f"Invalid number provided in range/count: '{text}'")
        return []

def show_spectrogram(spec: np.ndarray) -> None:
    """Renders a spectrogram without a title."""
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

# -------------------------------------------------------------------------
# Main UI Application
# -------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Mel-Spectrogram and Audio Viewer")

    # --- Sidebar (Controls) ---
    st.sidebar.header("Configuration")
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", options=sorted(DATASET_CONFIG.keys())
    )

    st.sidebar.subheader("Selection Method")
    range_text = st.sidebar.text_input(
        "Display by Range or Count",
        help="Priority input. E.g., '50' for first 50, or '50,150' for a range."
    )
    specific_indices_text = st.sidebar.text_input(
        "Display by Specific Indices",
        value="0 1 2 3 4",
        help="Fallback input. Used if the box above is empty. E.g., '5 10 15'."
    )

    display_btn = st.sidebar.button("Display", use_container_width=True, type="primary")

    if not display_btn:
        st.info("Configure your selection in the sidebar and click 'Display'.")
        st.stop()

    # --- Data Loading ---
    try:
        loader = get_loader(dataset_choice)
        num_available = len(loader)
        st.sidebar.success(f"Found {num_available} spectrograms for '{dataset_choice}'.\n\n"
                          f"Valid index range: 0 to {num_available - 1}.")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    # --- Index Parsing and Validation ---
    if range_text:
        all_requested_indices = parse_range_or_count(range_text)
    else:
        all_requested_indices = parse_specific_indices(specific_indices_text)

    valid_indices = [i for i in all_requested_indices if 0 <= i < num_available]
    invalid_indices = [i for i in all_requested_indices if not (0 <= i < num_available)]

    if invalid_indices:
        # Show a sample of invalid indices to avoid cluttering the UI
        st.warning(f"Indices out of range or invalid and were skipped: {invalid_indices[:10]}...")

    if not valid_indices:
        st.info("No valid samples to display based on your selection.")
        st.stop()

    # --- Large Request Warning ---
    if len(valid_indices) > 50:
        st.warning(
            f"Displaying {len(valid_indices)} samples. "
            "This may take a moment to load and will result in a long page."
        )

    # --- Main Panel (Dynamic Grid Display) ---
    st.header(f"Displaying {len(valid_indices)} Spectrogram(s) for: {dataset_choice.upper()}")

    cols = 5  # Number of columns in the grid
    # Create rows dynamically based on the number of valid indices
    for i in range(0, len(valid_indices), cols):
        # Get the indices for the current row
        row_indices = valid_indices[i:i + cols]
        # Create columns for the current row
        columns = st.columns(cols)

        # Populate each column with a spectrogram
        for col_index, spec_index in enumerate(row_indices):
            with columns[col_index]:
                try:
                    spec, filename, audio_path = loader.load(spec_index)
                    show_spectrogram(spec)
                    st.caption(f"Index: {spec_index} | File: {filename}")
                    if audio_path:
                        st.audio(str(audio_path), format="audio/wav")
                    else:
                        st.caption(f"Audio not found")
                except (IndexError, FileNotFoundError) as e:
                    st.error(f"Error loading index {spec_index}: {e}")

if __name__ == "__main__":
    main()