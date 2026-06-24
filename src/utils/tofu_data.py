"""TOFU dataset loading helpers."""

from datasets import load_dataset


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


def load_forget_split_entries(num_questions, split_name="forget10"):
    """
    Load question and ground-truth answer pairs from a TOFU forget split.

    Args:
        num_questions: Maximum number of entries to take from the split start.
        split_name: TOFU config name for the forget split.

    Returns:
        List of dicts with keys index, question, and ground_truth.
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_count = min(num_questions, len(dataset))
    return [
        {
            "index": index,
            "question": dataset[index]["question"],
            "ground_truth": dataset[index]["answer"],
        }
        for index in range(question_count)
    ]


def load_forget_split_dataset(num_questions, split_name="forget10"):
    """
    Load the TOFU forget split and return the dataset plus question count.

    Args:
        num_questions: Maximum number of questions to use from the split start.
        split_name: TOFU config name for the forget split.

    Returns:
        Tuple of (dataset, question_count).
    """
    dataset = load_dataset("locuslab/TOFU", split_name)["train"]
    question_count = min(num_questions, len(dataset))
    return dataset, question_count
