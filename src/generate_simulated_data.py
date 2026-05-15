from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "simulated_defect_fatigue_data.csv"
SEED = 42
N_SAMPLES = 260


def simulate_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    defect_type = rng.choice(
        ["gas_pore", "lack_of_fusion", "keyhole"],
        size=n_samples,
        p=[0.50, 0.35, 0.15],
    )
    build_orientation = rng.choice(
        ["XY", "XZ", "Z"],
        size=n_samples,
        p=[0.40, 0.35, 0.25],
    )
    heat_treatment = rng.choice(
        ["as_built", "stress_relieved"],
        size=n_samples,
        p=[0.58, 0.42],
    )
    material_family = rng.choice(
        ["Alloy_A", "Alloy_B"],
        size=n_samples,
        p=[0.62, 0.38],
    )

    defect_size_um = rng.lognormal(mean=4.8, sigma=0.45, size=n_samples)
    defect_size_um = np.clip(defect_size_um, 40.0, 420.0)

    depth_multiplier = rng.uniform(0.35, 3.50, size=n_samples)
    defect_depth_um = defect_size_um * depth_multiplier

    sphericity = np.clip(rng.beta(7.0, 2.6, size=n_samples), 0.35, 0.99)
    compactness = np.clip(rng.beta(6.0, 2.4, size=n_samples), 0.30, 0.99)
    aspect_ratio = np.clip(rng.lognormal(mean=0.18, sigma=0.40, size=n_samples), 0.70, 3.40)
    local_stress_mpa = np.clip(rng.normal(168.0, 18.0, size=n_samples), 115.0, 220.0)
    stress_ratio = np.clip(rng.normal(0.10, 0.05, size=n_samples), 0.02, 0.25)
    relative_density_pct = np.clip(rng.normal(99.25, 0.22, size=n_samples), 98.45, 99.85)

    type_factor = np.select(
        [defect_type == "gas_pore", defect_type == "lack_of_fusion", defect_type == "keyhole"],
        [0.00, 0.22, 0.12],
        default=0.0,
    )
    orientation_factor = np.select(
        [build_orientation == "XY", build_orientation == "XZ", build_orientation == "Z"],
        [0.00, 0.08, 0.13],
        default=0.0,
    )
    heat_treatment_factor = np.where(heat_treatment == "stress_relieved", -0.06, 0.03)
    material_factor = np.where(material_family == "Alloy_B", -0.05, 0.04)

    surface_proximity_ratio = defect_depth_um / np.maximum(defect_size_um, 1.0)
    surface_factor = np.where(surface_proximity_ratio < 0.90, 0.20, 0.0)

    stress_intensity_proxy = local_stress_mpa * np.sqrt(defect_size_um / 1000.0)
    morphology_penalty = 0.25 * (1.0 - sphericity) + 0.22 * (1.0 - compactness)
    density_penalty = 0.08 * (99.60 - relative_density_pct)

    log10_life = (
        6.65
        - 0.0028 * local_stress_mpa
        - 0.28 * np.log10(defect_size_um)
        - 0.18 * np.log10(np.maximum(stress_intensity_proxy, 1.0))
        - morphology_penalty
        - density_penalty
        - type_factor
        - orientation_factor
        - heat_treatment_factor
        - material_factor
        - surface_factor
        + rng.normal(0.0, 0.11, size=n_samples)
    )
    log10_life = np.clip(log10_life, 4.25, 6.90)
    fatigue_life_cycles = np.round(np.power(10.0, log10_life)).astype(int)

    df = pd.DataFrame(
        {
            "specimen_id": [f"SIM_{i:03d}" for i in range(1, n_samples + 1)],
            "material_family": material_family,
            "build_orientation": build_orientation,
            "heat_treatment": heat_treatment,
            "defect_type": defect_type,
            "defect_size_um": defect_size_um.round(2),
            "defect_depth_um": defect_depth_um.round(2),
            "sphericity": sphericity.round(3),
            "compactness": compactness.round(3),
            "aspect_ratio": aspect_ratio.round(3),
            "local_stress_mpa": local_stress_mpa.round(2),
            "stress_ratio": stress_ratio.round(3),
            "relative_density_pct": relative_density_pct.round(3),
            "fatigue_life_cycles": fatigue_life_cycles,
        }
    )

    # Inject a small amount of missingness to demonstrate cleaning/imputation.
    missing_rows = rng.choice(df.index, size=12, replace=False)
    df.loc[missing_rows[:4], "defect_depth_um"] = np.nan
    df.loc[missing_rows[4:8], "sphericity"] = np.nan
    df.loc[missing_rows[8:], "compactness"] = np.nan
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = simulate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

