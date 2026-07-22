from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .agent import HMATSADAgent
from .client import DashScopeClient
from .config import ExperimentConfig, load_experiment_config
from .data import load_sample
from .evaluation import evaluate_prediction_file, write_summary


def _resolve(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _dataset_sample_ids(data_root: Path, dataset: str) -> list[str]:
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    suffix = "_test.npy"
    sample_ids = sorted(path.name[: -len(suffix)] for path in dataset_dir.glob(f"*{suffix}"))
    if not sample_ids:
        raise ValueError(f"No test samples found in dataset: {dataset_dir}")
    return sample_ids


def _run_one(
    config: ExperimentConfig,
    project_root: Path,
    dataset: str,
    sample_id: str,
    force: bool,
) -> tuple[dict[str, Any], Path]:
    data_root = _resolve(config.data_root, project_root)
    output_root = _resolve(config.output_root, project_root)
    run_dir = output_root / config.agent.mode / dataset / sample_id
    prediction_path = run_dir / "predictions.json"
    if prediction_path.exists() and not force:
        return json.loads(prediction_path.read_text(encoding="utf-8")), prediction_path
    sample = load_sample(data_root, dataset, sample_id)
    client = DashScopeClient(config.model, project_root)
    agent = HMATSADAgent(client, config.agent, run_dir)
    result = agent.run(sample)
    return result, prediction_path


def command_run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    config = load_experiment_config(args.config, args.mode)
    if args.max_planner_steps is not None:
        config = replace(config, agent=replace(config.agent, max_planner_steps=args.max_planner_steps))
    result, path = _run_one(config, project_root, args.dataset, args.sample, args.force)
    print(json.dumps({"prediction_path": str(path), "result": result}, ensure_ascii=False, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    config = load_experiment_config(args.config)
    result = evaluate_prediction_file(args.prediction, _resolve(config.data_root, project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_experiment(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    config = load_experiment_config(args.config, args.mode)
    if args.max_planner_steps is not None:
        config = replace(config, agent=replace(config.agent, max_planner_steps=args.max_planner_steps))
    records: list[dict[str, Any]] = []
    data_root = _resolve(config.data_root, project_root)
    selected_dataset = args.dataset
    samples = (
        {selected_dataset: _dataset_sample_ids(data_root, selected_dataset)}
        if selected_dataset
        else config.samples
    )
    for dataset, sample_ids in samples.items():
        for sample_id in sample_ids:
            _, prediction_path = _run_one(config, project_root, dataset, sample_id, args.force)
            metrics = evaluate_prediction_file(prediction_path, data_root)
            records.append(metrics)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "mode": config.agent.mode,
                        "range_f1": metrics["range_f1"],
                        "point_f1": metrics["point_f1"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    output = _resolve(config.output_root, project_root) / config.agent.mode
    if selected_dataset:
        output /= selected_dataset
    summary = write_summary(records, output)
    print(json.dumps({"summary_path": str(output / "summary.json"), **summary}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="API-based HMA-TSAD reproduction")
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one inference sample")
    run.add_argument("--config", default="configs/fixed_subset.yaml")
    run.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    run.add_argument("--dataset", required=True)
    run.add_argument("--sample", required=True)
    run.add_argument("--max-planner-steps", type=int)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=command_run)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved prediction using labels")
    evaluate.add_argument("--config", default="configs/fixed_subset.yaml")
    evaluate.add_argument("--prediction", required=True)
    evaluate.set_defaults(func=command_evaluate)

    experiment = subparsers.add_parser("experiment", help="Run and evaluate configured samples or one full dataset")
    experiment.add_argument("--config", default="configs/fixed_subset.yaml")
    experiment.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    experiment.add_argument("--dataset", help="Run every *_test.npy sample in this dataset")
    experiment.add_argument("--max-planner-steps", type=int)
    experiment.add_argument("--force", action="store_true")
    experiment.set_defaults(func=command_experiment)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
