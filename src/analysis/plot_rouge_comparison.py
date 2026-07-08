"""Plot grouped bar charts comparing ROUGE-L recovery across models and steering."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROUGE_COMPARISON_RESULTS_DIR = Path("results/rouge-comparison")
ROUGE_THRESHOLD_OPTIONS = ("0.3", "0.6")

ROUGE_THRESHOLD_CONFIG = {
    "0.3": {
        "summary_key": "count_above_0.3",
        "ylabel": "Questions recovered (ROUGE-L > 0.3) out of 400",
        "title_suffix": "ROUGE-L > 0.3",
    },
    "0.6": {
        "summary_key": "count_above_0.6",
        "ylabel": "Questions recovered (ROUGE-L > 0.6) out of 400",
        "title_suffix": "ROUGE-L > 0.6",
    },
}

UNSTEERED_BAR_COLOR = "#4c78a8"
STEERED_BAR_COLOR = "#f58518"
BASELINE_BAR_COLOR = "#54a24b"


def default_rouge_comparison_plot_path(threshold):
    """
    Derive the default output path for a ROUGE comparison plot.

    Args:
        threshold: ROUGE threshold key, either "0.3" or "0.6".

    Returns:
        Path string such as results/rouge-comparison/rouge_comparison_0.3.png.
    """
    return str(ROUGE_COMPARISON_RESULTS_DIR / f"rouge_comparison_{threshold}.png")


def load_recovery_count(results_path, threshold):
    """
    Read a ROUGE recovery count from a probe or ablate-and-probe results JSON file.

    Args:
        results_path: Path to a results JSON file with a summary block.
        threshold: ROUGE threshold key, either "0.3" or "0.6".

    Returns:
        Integer count of questions with ROUGE-L above the requested threshold.

    Raises:
        KeyError: If the summary block has no usable count field for the threshold.
    """
    threshold_config = ROUGE_THRESHOLD_CONFIG[threshold]
    summary_key = threshold_config["summary_key"]
    results_path = Path(results_path)
    result_record = json.loads(results_path.read_text(encoding="utf-8"))
    summary = result_record["summary"]

    if summary_key in summary:
        return summary[summary_key]

    across_coefficients = summary.get("across_coefficients")
    if across_coefficients is not None and summary_key in across_coefficients:
        return across_coefficients[summary_key]

    raise KeyError(
        f"No {summary_key} found in summary for {results_path}. "
        "Expected a flat summary or across_coefficients summary."
    )


def plot_rouge_comparison_bars(
    idk_nll_unsteered_count,
    idk_nll_steered_count,
    npo_unsteered_count,
    npo_steered_count,
    baseline_count,
    output_path,
    threshold,
):
    """
    Plot grouped recovery counts for IDK-NLL, NPO, and baseline models.

    Args:
        idk_nll_unsteered_count: Recovery count for unsteered IDK-NLL.
        idk_nll_steered_count: Recovery count for steered IDK-NLL.
        npo_unsteered_count: Recovery count for unsteered NPO.
        npo_steered_count: Recovery count for steered NPO.
        baseline_count: Recovery count for the non-unlearned baseline model.
        output_path: Path to write the PNG plot.
        threshold: ROUGE threshold key, either "0.3" or "0.6".

    Returns:
        Path string for the saved plot file.
    """
    threshold_config = ROUGE_THRESHOLD_CONFIG[threshold]
    group_labels = ["IDK-NLL", "NPO", "Baseline"]
    group_positions = np.array([0.0, 1.0, 2.0])
    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        group_positions[0] - bar_width / 2,
        idk_nll_unsteered_count,
        bar_width,
        label="Unsteered",
        color=UNSTEERED_BAR_COLOR,
    )
    axis.bar(
        group_positions[0] + bar_width / 2,
        idk_nll_steered_count,
        bar_width,
        label="Steered",
        color=STEERED_BAR_COLOR,
    )
    axis.bar(
        group_positions[1] - bar_width / 2,
        npo_unsteered_count,
        bar_width,
        color=UNSTEERED_BAR_COLOR,
    )
    axis.bar(
        group_positions[1] + bar_width / 2,
        npo_steered_count,
        bar_width,
        color=STEERED_BAR_COLOR,
    )
    axis.bar(
        group_positions[2],
        baseline_count,
        bar_width,
        label="Baseline",
        color=BASELINE_BAR_COLOR,
    )

    axis.set_xticks(group_positions)
    axis.set_xticklabels(group_labels)
    axis.set_ylabel(threshold_config["ylabel"])
    axis.set_title(
        f"ROUGE-L recovery comparison ({threshold_config['title_suffix']})"
    )
    axis.legend()
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved ROUGE comparison plot to {output_path}", flush=True)
    return str(output_path)


def plot_rouge_comparison(
    idk_nll_unsteered_path,
    idk_nll_steered_path,
    npo_unsteered_path,
    npo_steered_path,
    baseline_path,
    output_path,
    threshold="0.3",
):
    """
    Load probe and ablation result files and write a grouped recovery bar chart.

    Args:
        idk_nll_unsteered_path: Path to unsteered IDK-NLL probe JSON.
        idk_nll_steered_path: Path to steered IDK-NLL ablate-and-probe JSON.
        npo_unsteered_path: Path to unsteered NPO probe JSON.
        npo_steered_path: Path to steered NPO ablate-and-probe JSON.
        baseline_path: Path to baseline probe JSON.
        output_path: Path to write the PNG plot.
        threshold: ROUGE threshold to plot, either "0.3" or "0.6".

    Returns:
        Path string for the saved plot file.

    Raises:
        ValueError: If threshold is not a supported option.
    """
    if threshold not in ROUGE_THRESHOLD_OPTIONS:
        supported = ", ".join(ROUGE_THRESHOLD_OPTIONS)
        raise ValueError(f"Unsupported threshold '{threshold}'. Choose from: {supported}.")

    return plot_rouge_comparison_bars(
        load_recovery_count(idk_nll_unsteered_path, threshold),
        load_recovery_count(idk_nll_steered_path, threshold),
        load_recovery_count(npo_unsteered_path, threshold),
        load_recovery_count(npo_steered_path, threshold),
        load_recovery_count(baseline_path, threshold),
        output_path,
        threshold,
    )
