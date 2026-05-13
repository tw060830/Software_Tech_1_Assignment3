from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def save_training_report(results: dict[str, object], output_dir: Path) -> None:
    """Write the classification report to a text file."""

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(results["report"], encoding="utf-8")

def save_confusion_matrix_plot(
    results: dict[str, object],
    labels: list[str],
    output_dir: Path,) -> None:
    """Save a confusion matrix heatmap image."""

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results["confusion_matrix"],
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()