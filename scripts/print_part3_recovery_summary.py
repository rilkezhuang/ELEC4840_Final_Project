"""Print Recovery-toward-tight-box table from part3_box_recovery.csv (for report / notebook)."""
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECOVERY_CSV = os.path.join(PROJECT_ROOT, "outputs", "part3_box_recovery.csv")


def main():
    if not os.path.exists(RECOVERY_CSV):
        print(f"Missing {RECOVERY_CSV}. Run: python scripts/eval_part3_box_recovery.py")
        sys.exit(1)

    rec = pd.read_csv(RECOVERY_CSV)
    gap_df = rec.groupby("Task")[
        ["Dice_Tight_Oracle", "Dice_Loose_Benchmark", "Dice_PRA_Refined"]
    ].mean()
    gap_df.columns = [
        "Tight Box (oracle)",
        "Loose Box (Part II +10px)",
        "Loose + PRA (benchmark loose)",
    ]
    gap_df["PRA vs Loose"] = (
        gap_df["Loose + PRA (benchmark loose)"] - gap_df["Loose Box (Part II +10px)"]
    )
    gap_df["Gap to Tight remaining"] = (
        gap_df["Tight Box (oracle)"] - gap_df["Loose + PRA (benchmark loose)"]
    )

    print("=== Recovery toward Tight Box (benchmark loose box, mean Dice) ===")
    print(gap_df.round(3).to_string())
    print()

    box_rec = rec.groupby("Task")[
        ["Loose_Box_IoU_to_Tight", "Refined_Box_IoU_to_Tight", "Box_IoU_Recovery"]
    ].mean()
    box_rec.columns = ["Loose IoU→Tight", "Refined IoU→Tight", "IoU gain"]
    print("=== Box geometry recovery (vs. GT tight box) ===")
    print(box_rec.round(4).to_string())


if __name__ == "__main__":
    main()
