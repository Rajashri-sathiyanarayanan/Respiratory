import csv
from pathlib import Path
from typing import Any, Dict, List

import librosa

from src.config import PATHS
from src.ontology.schema import (
    CoughAudioRef,
    ClinicalState,
    Diagnosis,
    PatientSample,
    Symptom,
)


RESP_SOUNDS_ROOT = PATHS.project_root / "data" / "raw" / "resp_sounds"
AUDIO_TXT_DIR = RESP_SOUNDS_ROOT / "audio_and_txt_files"
DIAG_CSV = RESP_SOUNDS_ROOT / "patient_diagnosis.csv"


def load_patient_diagnoses() -> Dict[str, Dict[str, Any]]:
    """
    Load patient diagnosis metadata keyed by recording filename.
    """
    diag_path = DIAG_CSV
    mapping: Dict[str, Dict[str, Any]] = {}
    with diag_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The original dataset often uses a 'filename' or 'Record' column
            fname = row.get("filename") or row.get("Record") or row.get("recordname")
            if not fname:
                continue
            mapping[fname] = row
    return mapping


def parse_segment_file(txt_path: Path) -> List[Symptom]:
    """
    Parse a .txt segment file with lines: start end crackle_flag wheeze_flag
    We convert crackle/wheeze flags into pseudo-symptoms.
    """
    symptoms: List[Symptom] = []
    try:
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                _, _, crackle_flag, wheeze_flag = parts
                if crackle_flag == "1":
                    symptoms.append(Symptom(code="crackle"))
                if wheeze_flag == "1":
                    symptoms.append(Symptom(code="wheeze"))
    except FileNotFoundError:
        # Some files may not have a corresponding txt; that's fine.
        pass
    return symptoms


def load_audio_metadata(file_path: Path) -> CoughAudioRef:
    """
    Load minimal metadata for an audio file (duration, sample rate).
    """
    y, sr = librosa.load(file_path, sr=None)
    duration = float(len(y) / sr) if sr else 0.0
    return CoughAudioRef(
        file_path=str(file_path.relative_to(PATHS.project_root)),
        sample_rate=int(sr),
        duration_sec=duration,
    )


def build_resp_sound_samples() -> List[PatientSample]:
    """
    Create PatientSample objects from the Respiratory Sound Database.
    """
    diag_map = load_patient_diagnoses()
    samples: List[PatientSample] = []

    for wav_path in AUDIO_TXT_DIR.glob("*.wav"):
        base = wav_path.stem  # e.g. '101_1b1_Al_sc_Meditron'
        txt_path = AUDIO_TXT_DIR / f"{base}.txt"

        # Symptoms from crackle/wheeze annotations
        symptoms = parse_segment_file(txt_path)

        # Diagnosis metadata
        diag_row = diag_map.get(f"{base}.wav") or diag_map.get(base)
        if diag_row:
            diagnosis_label = diag_row.get("diagnosis") or diag_row.get("class")
        else:
            diagnosis_label = None

        diagnosis = Diagnosis(label=diagnosis_label) if diagnosis_label else None

        # Clinical state is unknown; we keep it None for now
        clinical_state = ClinicalState(state_label="unknown")

        audio_ref = load_audio_metadata(wav_path)

        sample = PatientSample(
            sample_id=f"resp_sounds:{base}",
            source_dataset="Respiratory_Sound_Database",
            patient_id=base.split("_")[0] if "_" in base else None,
            timestamp=None,
            demographics={},
            symptoms=symptoms,
            spirometry=None,
            cough_audio=audio_ref,
            diagnosis=diagnosis,
            clinical_state=clinical_state,
        )
        samples.append(sample)

    return samples


if __name__ == "__main__":
    out_dir = PATHS.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_resp_sound_samples()

    # For now, we just print count; later we will serialize to parquet/json
    print(f"Built {len(samples)} PatientSample objects from Respiratory Sound Database")








