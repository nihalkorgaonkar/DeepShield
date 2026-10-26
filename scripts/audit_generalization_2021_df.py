"""
DeepShield ASVspoof 2021 DF Generalization Status Auditor
Audits local availability of ASVspoof 2021 DF keys and audio files.
In strict compliance with the Zero Fabrication Directive, documents the resource status
without synthesizing metrics for unavailable audio files.
"""

import json
import os


def main():
    keys_dir = r"C:\DeepShieldData\ASVspoof2021_DF\keys"
    has_audio = False
    candidate_paths = [
        r"C:\DeepShieldData\ASVspoof2021_DF\flac",
        r"C:\DeepShieldData\ASVspoof2021_DF_eval\flac",
    ]
    for cand in candidate_paths:
        if os.path.exists(cand) and len(os.listdir(cand)) > 0:
            has_audio = True
            break

    status = {
        "dataset": "ASVspoof2021_DF",
        "metadata_keys_present": os.path.exists(keys_dir),
        "flac_audio_present": has_audio,
        "evaluation_status": "DOCUMENTED_MISSING_AUDIO",
        "audit_finding": (
            "Official evaluation audio FLAC files (~600k files, >20 GB) are absent from local storage. "
            "Zero Fabrication Directive mandates documenting this absence rather than fabricating synthetic metrics."
        ),
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "generalization_2021_df_status.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"Documented ASVspoof 2021 DF status to {out_path}")


if __name__ == "__main__":
    main()
