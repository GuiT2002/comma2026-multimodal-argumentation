#!/usr/bin/env python3
"""Print descriptive statistics for the binary evaluations in validation CSVs."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CSV_FIELDS = (
    "expected_output",
    "model_output",
    "scheme_correct",
    "semantic_match",
)
CLASS_LABELS = {
    "class1": "Geographic",
    "class2": "Temporal",
    "class3": "Geographic + Temporal",
}
EXPERIMENTS = {
    1: ("without_enthymeme", "Image + Context"),
    2: ("without_context", "Enthymeme + Image"),
    3: ("without_image", "Enthymeme + Context"),
    4: ("all_inputs", "Enthymeme + Image + Context"),
}
FILE_PATTERN = re.compile(
    r"(?P<model>.+)_(?P<class_name>class[1-3])_col(?P<column>[1-4])\.csv"
)
OUTCOMES = (("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no"))


@dataclass
class ValidationBatch:
    model: str
    class_name: str
    column: int
    outcomes: Counter


def read_outcomes(stream, source):
    """Validate an entire CSV and count its four possible binary outcomes."""
    reader = csv.reader(stream, strict=True)
    outcomes = Counter()
    try:
        header = next(reader, None)
        if header != list(CSV_FIELDS):
            raise ValueError(
                f"{source}: expected the CSV header {','.join(CSV_FIELDS)}."
            )
        for row in reader:
            if len(row) != len(CSV_FIELDS):
                raise ValueError(
                    f"{source}, line {reader.line_num}: expected four columns, "
                    f"found {len(row)}."
                )
            for field, value in zip(CSV_FIELDS[2:], row[2:]):
                if value not in ("yes", "no"):
                    raise ValueError(
                        f"{source}, line {reader.line_num}: {field} must be "
                        f"exactly 'yes' or 'no'; found {value!r}."
                    )
            outcomes[(row[2], row[3])] += 1
    except csv.Error as exc:
        raise ValueError(
            f"{source}, line {reader.line_num}: invalid CSV: {exc}"
        ) from exc
    if not outcomes:
        raise ValueError(f"{source}: the CSV contains no evaluations.")
    return outcomes


def load_validation_batches(validation_dir):
    """Load and validate all evaluation CSVs before printing any statistics."""
    if not validation_dir.is_dir():
        raise ValueError(f"Validation directory does not exist: {validation_dir}")
    paths = sorted(validation_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"No validation CSV files found in {validation_dir}.")
    batches = []
    for path in paths:
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(
                f"{path}: expected a filename such as "
                "claude-sonnet_class1_col1.csv "
                "(classes 1-3, columns 1-4)."
            )
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            outcomes = read_outcomes(stream, path)
        batches.append(
            ValidationBatch(
                model=match["model"],
                class_name=match["class_name"],
                column=int(match["column"]),
                outcomes=outcomes,
            )
        )
    return batches


def format_rate(numerator, denominator):
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.2f}%)"


def metric_cells(outcomes):
    total = sum(outcomes.values())
    scheme_yes = sum(count for (scheme, _), count in outcomes.items() if scheme == "yes")
    semantic_yes = sum(count for (_, semantic), count in outcomes.items() if semantic == "yes")
    return [
        str(total),
        format_rate(scheme_yes, total),
        format_rate(semantic_yes, total),
        format_rate(outcomes[("yes", "yes")], total),
    ]


def print_table(title, headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    print(f"\n{title}")
    print(" | ".join(str(value).ljust(width) for value, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))


def print_statistics(batches, validation_dir):
    total_outcomes = Counter()
    for batch in batches:
        total_outcomes.update(batch.outcomes)
    total = sum(total_outcomes.values())
    print(f"Validation directory: {validation_dir}")
    print(f"CSV files: {len(batches)}")
    print(f"Evaluations: {total}")
    print("Rates are micro-averaged: yes count / evaluations in each group.")
    print("Each evaluation is one case in one model/class/experiment CSV.")
    print("Statistics cover the files present; incomplete runs are not extrapolated.")
    print("Classes: " + "; ".join(f"{key} = {label}" for key, label in CLASS_LABELS.items()))
    print("Experiments:")
    for column, (name, inputs) in EXPERIMENTS.items():
        print(f"  col{column} = {name}: {inputs}")

    metric_headers = ["Evaluations", "Scheme correct", "Semantic match", "Both yes"]
    print_table("Overall", metric_headers, [metric_cells(total_outcomes)])
    print_table(
        "Joint outcomes",
        ["Scheme correct", "Semantic match", "Count / evaluations (rate)"],
        [[scheme, semantic, format_rate(total_outcomes[(scheme, semantic)], total)]
         for scheme, semantic in OUTCOMES],
    )

    groupings = (
        ("By model", ("Model",), ("model",)),
        ("By class", ("Class",), ("class_name",)),
        ("By experiment", ("Experiment",), ("column",)),
        ("By model and class", ("Model", "Class"), ("model", "class_name")),
        ("By model and experiment", ("Model", "Experiment"), ("model", "column")),
        ("By class and experiment", ("Class", "Experiment"), ("class_name", "column")),
    )
    for title, headers, attributes in groupings:
        groups = defaultdict(Counter)
        for batch in batches:
            key = tuple(getattr(batch, attribute) for attribute in attributes)
            groups[key].update(batch.outcomes)
        rows = []
        for key, outcomes in sorted(groups.items()):
            labels = [
                f"col{value} ({EXPERIMENTS[value][0]})" if attribute == "column" else value
                for attribute, value in zip(attributes, key)
            ]
            rows.append(labels + metric_cells(outcomes))
        print_table(title, [*headers, *metric_headers], rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print binary validation statistics grouped by model, class, and experiment."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=BASE_DIR / "validation",
        help="Directory containing validation CSVs (default: validation next to this script).",
    )
    args = parser.parse_args(argv)
    try:
        batches = load_validation_batches(args.validation_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print_statistics(batches, args.validation_dir)


if __name__ == "__main__":
    main()
