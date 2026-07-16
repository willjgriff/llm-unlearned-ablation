"""TOFU dataset loading helpers."""

from datasets import load_dataset

from utils.constants import (
    FORGET_PERTURBED_SPLIT,
    FORGET_SPLIT,
    QUESTION_MODE_ORIGINAL,
    QUESTION_MODE_PERTURBED,
)


def resolve_forget_split_name(question_mode=QUESTION_MODE_ORIGINAL, split_name=None):
    """
    Resolve the TOFU Hugging Face config name for a forget-set probe run.

    Args:
        question_mode: Either original (exact training questions) or perturbed
            (paraphrased questions from forget10_perturbed).
        split_name: Optional explicit split override; when omitted, inferred from
            question_mode.

    Returns:
        TOFU dataset config name (e.g. forget10 or forget10_perturbed).
    """
    if split_name is not None:
        return split_name
    if question_mode == QUESTION_MODE_PERTURBED:
        return FORGET_PERTURBED_SPLIT
    return FORGET_SPLIT


def question_field_for_mode(question_mode):
    """
    Return the dataset column to use as the probe prompt for a question mode.

    Args:
        question_mode: Either original or perturbed.

    Returns:
        Column name on the TOFU dataset row (question or paraphrased_question).
    """
    if question_mode == QUESTION_MODE_PERTURBED:
        return "paraphrased_question"
    return "question"


def load_questions(split_name, num_questions):
    """
    Load question strings from a TOFU dataset split.

    Args:
        split_name: TOFU config name (e.g. forget10 or retain90).
        num_questions: Maximum number of questions to take from the split start.

    Returns:
        List of question strings.
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_count = min(num_questions, len(dataset))
    return [dataset[index]["question"] for index in range(question_count)]


def load_forget_split_entries(
    num_questions,
    split_name=FORGET_SPLIT,
    question_mode=QUESTION_MODE_ORIGINAL,
):
    """
    Load question and ground-truth answer pairs from a TOFU forget split.

    Args:
        num_questions: Maximum number of entries to take from the split start.
        split_name: TOFU config name for the forget split.
        question_mode: Whether to use original or paraphrased question text.

    Returns:
        List of dicts with keys index, question, and ground_truth.
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_field = question_field_for_mode(question_mode)
    question_count = min(num_questions, len(dataset))
    return [
        {
            "index": index,
            "question": dataset[index][question_field],
            "ground_truth": dataset[index]["answer"],
        }
        for index in range(question_count)
    ]


def load_forget_split_dataset(
    num_questions,
    split_name=FORGET_SPLIT,
    question_mode=QUESTION_MODE_ORIGINAL,
):
    """
    Load the TOFU forget split and return the dataset plus question count.

    Args:
        num_questions: Maximum number of questions to use from the split start.
        split_name: TOFU config name for the forget split.
        question_mode: Whether to use original or paraphrased question text.

    Returns:
        Tuple of (dataset, question_count, question_field).
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_count = min(num_questions, len(dataset))
    question_field = question_field_for_mode(question_mode)
    return dataset, question_count, question_field
