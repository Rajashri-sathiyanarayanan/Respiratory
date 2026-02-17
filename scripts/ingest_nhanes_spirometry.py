import pandas as pd
from pathlib import Path
from typing import List

from src.config import PATHS
from src.ontology.schema import ClinicalState, Diagnosis, PatientSample, SpirometryMeasure


NHANES_DIR = PATHS.project_root / "data" / "raw" / "nhanes_respiratory"
NHANES_CSV = NHANES_DIR / "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"


def derive_diagnosis_and_state(row: pd.Series) -> (Diagnosis, ClinicalState):
    """
    Derive obstructive pattern and coarse clinical state from spirometry.
    This is a simplified interpretation of GOLD-like rules.
    """
    ratio = row.get("FEV1_FVC_Ratio")
    fev1 = row.get("FEV1")
    pred_fev1 = row.get("Predicted_FEV1_Linear") or row.get("Predicted_FEV1_Segmented")

    diagnosis_label = "Normal"
    state_label = "stable"

    if pd.notnull(ratio) and ratio < 0.7:
        diagnosis_label = "Obstructive"
        # Derive severity from % predicted FEV1 if available
        if pd.notnull(fev1) and pd.notnull(pred_fev1) and pred_fev1 > 0:
            pct = 100.0 * fev1 / pred_fev1
            if pct >= 80:
                state_label = "mild_obstruction"
            elif pct >= 50:
                state_label = "moderate_obstruction"
            elif pct >= 30:
                state_label = "severe_obstruction"
            else:
                state_label = "very_severe_obstruction"
        else:
            state_label = "obstructive_pattern"

    diagnosis = Diagnosis(label=diagnosis_label)
    clinical_state = ClinicalState(state_label=state_label)
    return diagnosis, clinical_state


def build_nhanes_samples() -> List[PatientSample]:
    df = pd.read_csv(NHANES_CSV)
    samples: List[PatientSample] = []

    for _, row in df.iterrows():
        spirometry = SpirometryMeasure(
            fev1=row.get("FEV1"),
            fvc=row.get("FVC"),
            fev1_fvc_ratio=row.get("FEV1_FVC_Ratio"),
            pef=row.get("PEF"),
        )

        diagnosis, clinical_state = derive_diagnosis_and_state(row)

        demographics = {
            "sex": str(row.get("Sex")),
            "race": str(row.get("Race")),
            "age": str(row.get("Age")),
            "bmi": str(row.get("BMI")),
        }

        seqn = str(row.get("SEQN"))

        sample = PatientSample(
            sample_id=f"nhanes:{seqn}",
            source_dataset="NHANES_2007_2012_Spirometry",
            patient_id=seqn,
            timestamp=None,
            demographics=demographics,
            symptoms=[],
            spirometry=spirometry,
            cough_audio=None,
            diagnosis=diagnosis,
            clinical_state=clinical_state,
        )
        samples.append(sample)

    return samples


if __name__ == "__main__":
    out_dir = PATHS.data_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_nhanes_samples()
    print(f"Built {len(samples)} PatientSample objects from NHANES spirometry")








