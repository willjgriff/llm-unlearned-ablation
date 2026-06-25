"""Text similarity metrics for probe result evaluation."""

from rouge_score import rouge_scorer

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(reference_text, candidate_text):
    """
    Compute ROUGE-L between a reference answer and a model-generated answer.

    Args:
        reference_text: Ground-truth answer text from the dataset.
        candidate_text: Model-generated answer text.

    Returns:
        ROUGE-L F1 score in the range [0.0, 1.0].
    """
    rouge_scores = _ROUGE_SCORER.score(reference_text, candidate_text)
    return float(rouge_scores["rougeL"].fmeasure)


def summarize_rouge_scores(scores):
    """
    Aggregate ROUGE-L scores into mean and threshold counts.

    Args:
        scores: Iterable of numeric ROUGE-L scores.

    Returns:
        Dict with mean_rouge_l, count_above_0.3, and count_above_0.6.
    """
    score_list = list(scores)
    if not score_list:
        return {
            "mean_rouge_l": 0.0,
            "count_above_0.3": 0,
            "count_above_0.6": 0,
        }
    return {
        "mean_rouge_l": sum(score_list) / len(score_list),
        "count_above_0.3": sum(1 for score in score_list if score > 0.3),
        "count_above_0.6": sum(1 for score in score_list if score > 0.6),
    }


def summarize_flat_results(results):
    """
    Build a summary block from flat per-question results with rouge_l scores.

    Args:
        results: List of per-question dicts each containing a rouge_l field.

    Returns:
        Summary dict with mean_rouge_l and threshold counts.
    """
    return summarize_rouge_scores(result["rouge_l"] for result in results)


def summarize_sweep_results(coefficient_runs, grouped_results):
    """
    Build a sweep summary with per-coefficient stats and across-coefficient counts.

    Args:
        coefficient_runs: List of dicts with steering_coefficient and results keys.
        grouped_results: Per-question results grouped by coefficient, each multi-coef
            entry containing model_answers and max_rouge_l.

    Returns:
        Summary dict with per_coefficient and across_coefficients sections.
    """
    per_coefficient = []
    for coefficient_run in coefficient_runs:
        coefficient_summary = summarize_rouge_scores(
            result["rouge_l"] for result in coefficient_run["results"]
        )
        per_coefficient.append(
            {
                "steering_coefficient": coefficient_run["steering_coefficient"],
                **coefficient_summary,
            }
        )

    max_scores = [result["max_rouge_l"] for result in grouped_results]
    return {
        "per_coefficient": per_coefficient,
        "across_coefficients": {
            "count_above_0.3": sum(1 for score in max_scores if score > 0.3),
            "count_above_0.6": sum(1 for score in max_scores if score > 0.6),
        },
    }


def attach_summary_before_results(run_record, summary):
    """
    Insert summary immediately before results in a run record dict.

    Args:
        run_record: Mutable dict containing a results key.
        summary: Summary dict to insert.

    Returns:
        The same run_record dict with summary placed before results.
    """
    results = run_record.pop("results")
    run_record["summary"] = summary
    run_record["results"] = results
    return run_record
