"""Generate placeholder PNGs for the three figures so LaTeX builds."""
import matplotlib.pyplot as plt
import os

ROOT = r"c:\Users\Abhishek Soni\OneDrive\Desktop\FINAL_FYP"
FIG = os.path.join(ROOT, "figures")

targets = {
    "calibration_diagram.png": (
        "FIGURE NOT AVAILABLE",
        "calibration_diagram.png",
        "Run: python scratch/_temp_fig1_calibration.py\n"
        "Script could not be executed in this environment\n"
        "(Python execution blocked by sandbox)."
    ),
    "training_curves.png": (
        "FIGURE NOT AVAILABLE",
        "training_curves.png",
        "Run: python scratch/_temp_fig2_training_curves.py\n"
        "Script could not be executed in this environment\n"
        "(Python execution blocked by sandbox)."
    ),
    "decision_boundary.png": (
        "FIGURE NOT AVAILABLE",
        "decision_boundary.png",
        "Run: python scratch/_temp_fig3_decision_boundary.py\n"
        "Script could not be executed in this environment\n"
        "(Python execution blocked by sandbox)."
    ),
}

for fname, (title, sub, body) in targets.items():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#c0392b'); s.set_linewidth(2)
    ax.text(0.5, 0.72, title, ha='center', va='center',
            fontsize=22, fontweight='bold', color='#c0392b',
            transform=ax.transAxes)
    ax.text(0.5, 0.55, sub, ha='center', va='center',
            fontsize=13, fontweight='bold', color='#2c3e50',
            transform=ax.transAxes)
    ax.text(0.5, 0.28, body, ha='center', va='center',
            fontsize=10, color='#34495e', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("wrote", fname)
