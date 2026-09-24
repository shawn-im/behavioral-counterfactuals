#!/usr/bin/env python3
from __future__ import annotations

"""Run the staged applied-edit counterfactual analysis harness."""

import argparse
import gc
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])

import torch

# Bound PyTorch's host-side pools as well as BLAS/OpenMP.  The default is two
# threads per process, configurable with FAITHFUL_INTERVENTIONS_CPU_THREADS.
try:
    _cpu_threads = max(1, int(__import__("os").environ.get("FAITHFUL_INTERVENTIONS_CPU_THREADS", "2")))
    torch.set_num_threads(_cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except (TypeError, ValueError):
    _cpu_threads = 2

from behavior_pipeline.counterfactual_focused_analysis import (
    build_keyword_sentence_browser,
    compute_exhaustive_edit_event_validation,
    merge_exhaustive_validation_shards,
    partition_record_paths,
    render_exhaustive_validation_html,
    render_sentence_browser_html,
    select_random_record_paths,
)
from behavior_pipeline.counterfactual_harness import STAGE_NAMES, StageRunner, load_config, override_model_devices, path_fingerprint, prune_generated_directory, sha256_file, stage_seed, topological_stage_order, write_json, write_yaml
from behavior_pipeline.counterfactual_report import build_report
from behavior_pipeline.native_hypothesis import (
    build_native_hypothesis_report,
    compute_native_hypothesis_analysis,
    compute_native_hypothesis_analysis_shard,
    load_native_runtime,
    merge_native_hypothesis_shards,
    partition_summaries_by_source,
    select_native_record_summaries,
)
from behavior_pipeline.counterfactual_runtime import compute_scalar_kl_attribution, prepare_attribution_record_shards, prepare_ablation_record_shards, compute_edit_event_ablations, load_record_summaries, load_runtime, merge_attribution_shards, merge_teacher_forced_shards, merge_edit_event_ablation_shards, trace_teacher_forced, select_qualitative_record_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resumable applied-edit counterfactual analysis harness."
    )
    parser.add_argument("--config", required=True, help="YAML harness configuration")
    parser.add_argument("--checkpoint", help="Override edit.checkpoint")
    parser.add_argument("--cache_dir", help="Override data.cache_dir")
    parser.add_argument("--output_dir", help="Override run.output_dir")
    parser.add_argument(
        "--max_examples",
        type=int,
        help="Override the analysis example limit (0 means all)",
    )
    parser.add_argument(
        "--device",
        "--devices",
        dest="devices",
        nargs="+",
        help="Override YAML devices; accepts space/comma-separated devices.",
    )
    parser.add_argument(
        "--stages",
        default="0-7",
        help="Stage selection: 4=native patching, 5=exhaustive validation, 6=sentence browser, 7=report. Default: 0-7",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected stages even when resumability hashes match",
    )
    return parser.parse_args()


def parse_stages(value: str) -> list[int]:
    selected = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = [index for index in selected if index < 0 or index >= len(STAGE_NAMES)]
    if invalid:
        raise ValueError(f"Invalid stage numbers: {invalid}")
    return sorted(selected)


def validate_checkpoint_feedthrough(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    config = dict(payload.get("config", {}))
    edit_kind = str(config.get("edit_kind", ""))
    if edit_kind != "base_mlp":
        raise RuntimeError(
            "counterfactual interpretation is defined for the primary base_mlp edit module; "
            f"got edit_kind={edit_kind!r}"
        )
    return {
        "edit_kind": edit_kind,
        "intermediate_size": int(config.get("edit_mlp_hidden_size", 0)),
        "initialization": str(config.get("initialization", "")),
    }


def report_external_inputs(config: dict) -> tuple[Path, ...]:
    # Track missing configured files too: their later creation must invalidate
    # a completed report, just as changing or deleting their contents does.
    return tuple(
        Path(str(value))
        for value in (
            config.get("data", {}).get("static_runs_dir", ""),
            config.get("report", {}).get("objective_comparison_path", ""),
        )
        if value is not None and str(value).strip() not in {"", "."}
    )


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.checkpoint:
        overrides.setdefault("edit", {})["checkpoint"] = str(args.checkpoint)
    if args.cache_dir:
        overrides.setdefault("data", {})["cache_dir"] = str(args.cache_dir)
    if args.output_dir:
        overrides.setdefault("run", {})["output_dir"] = str(args.output_dir)
    if args.max_examples is not None:
        overrides.setdefault("data", {})["max_examples"] = int(args.max_examples)
    config = load_config(args.config, overrides=overrides)
    override_model_devices(config, args.devices)
    selected = parse_stages(args.stages)
    root = Path(__file__).resolve().parents[1]
    run_dir = Path(config["run"]["output_dir"])
    runner = StageRunner(config, root=root, force=args.force)
    runtime_holder = {}

    def configured_devices() -> list[str]:
        devices = [str(device) for device in config["model"].get("devices", [])]
        primary = str(config["model"]["device"])
        if not devices:
            return [primary]
        return list(dict.fromkeys([primary, *devices]))

    print(
        "[harness] execution: "
        f"devices={','.join(configured_devices())} "
        f"teacher_batch={int(config['teacher_forced'].get('batch_size', 1))} "
        f"ig_integration_batch={int(config['attribution'].get('integration_batch_size', 1))} "
        f"ablation_batch={int(config['ablations'].get('batch_size', 1))} "
        f"cpu_threads={_cpu_threads}"
    )

    def runtime():
        if "runtime" not in runtime_holder:
            runtime_holder["runtime"] = load_runtime(config)
        return runtime_holder["runtime"]

    def runtimes():
        if "runtimes" not in runtime_holder:
            values = [runtime()]
            for device in configured_devices()[1:]:
                values.append(load_runtime(config, device_override=device))
            runtime_holder["runtimes"] = values
        return runtime_holder["runtimes"]

    def native_runtimes():
        if "native_runtimes" not in runtime_holder:
            values = [load_native_runtime(config, device_override=configured_devices()[0])]
            for device in configured_devices()[1:]:
                values.append(load_native_runtime(config, device_override=device))
            runtime_holder["native_runtimes"] = values
        return runtime_holder["native_runtimes"]

    def release_edit_runtimes():
        # A full invocation may have loaded one base-model+editor runtime per
        # GPU for attribution/ablations. Stage 4 uses editor-free
        # runtimes, so release the earlier copies before loading another model.
        runtime_holder.pop("runtimes", None)
        runtime_holder.pop("runtime", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_native_runtimes():
        runtime_holder.pop("native_runtimes", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stage_functions = {}

    def stage_manifest():
        checkpoint = Path(config["edit"]["checkpoint"])
        checkpoint_info = validate_checkpoint_feedthrough(checkpoint)
        resolved = {key: value for key, value in config.items() if not key.startswith("_")}
        write_yaml(run_dir / "config.yaml", resolved)
        manifest = {
            "analysis_object": "counterfactual_applied_edit_events",
            "run_name": config["run"]["name"],
            "config_source": config["_config_path"],
            "model": path_fingerprint(Path(config["model"]["name"])),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                **checkpoint_info,
            },
            "cache": path_fingerprint(Path(config["data"]["cache_dir"])),
        }
        write_json(run_dir / "manifest.json", manifest)
        return [run_dir / "manifest.json", run_dir / "config.yaml"]

    stage_functions[0] = (
        stage_manifest,
        (),
        ("run", "model", "edit", "data"),
        (
            Path(config["model"]["name"]),
            Path(config["edit"]["checkpoint"]),
            Path(config["data"]["cache_dir"]),
        ),
    )

    def stage_teacher_forced():
        output = run_dir / "teacher_forced"
        if not bool(config["teacher_forced"]["enabled"]):
            write_json(output / "summary.json", {"status": "disabled", "records": 0})
            summary = {"status": "disabled", "records": 0}
        else:
            with stage_seed(config, 1):
                runtime_values = runtimes()
                if len(runtime_values) == 1:
                    summary = trace_teacher_forced(
                        config, runtime_values[0], output
                    )
                else:
                    with tempfile.TemporaryDirectory(
                        prefix=".stage1_shards_", dir=run_dir
                    ) as temporary:
                        shard_root = Path(temporary)
                        shard_directories = [
                            shard_root / f"worker_{index}"
                            for index in range(len(runtime_values))
                        ]

                        def teacher_worker(worker_index: int):
                            return trace_teacher_forced(
                                config,
                                runtime_values[worker_index],
                                shard_directories[worker_index],
                                shard_index=worker_index,
                                shard_count=len(runtime_values),
                            )

                        with ThreadPoolExecutor(
                            max_workers=len(runtime_values)
                        ) as executor:
                            futures = [
                                executor.submit(teacher_worker, index)
                                for index in range(len(runtime_values))
                            ]
                            for future in futures:
                                future.result()
                        summary = merge_teacher_forced_shards(
                            shard_directories,
                            output,
                            minimum_source_kl=float(
                                config["teacher_forced"].get(
                                    "min_source_kl", 0.0
                                )
                            ),
                        )
        warnings = []
        if summary.get("support_mass_p10") is not None and float(summary["support_mass_p10"]) < 0.8:
            warnings.append(
                f"Teacher-forced support mass p10 is low: {float(summary['support_mass_p10']):.6f}"
            )
        return {"outputs": [output], "warnings": warnings}

    stage_functions[1] = (
        stage_teacher_forced,
        (0,),
        ("model", "edit", "data", "teacher_forced"),
        (),
    )

    def record_directories() -> dict[str, Path]:
        return {"teacher_forced": run_dir / "teacher_forced"}

    def stage_attribution():
        output = run_dir / "attribution"
        output.mkdir(parents=True, exist_ok=True)
        methods = tuple(str(value) for value in config["attribution"].get("methods", ["ig"]))
        if bool(config["attribution"]["enabled"]):
            with stage_seed(config, 3):
                directories = record_directories()
                runtime_values = runtimes()
                # Plan all GPU shards once from lightweight JSON sidecars.  The
                # workers then stream their assigned .pt traces one at a time.
                attribution_record_plans = prepare_attribution_record_shards(
                    config,
                    directories,
                    shard_count=len(runtime_values),
                )
                for mode in directories:
                    counts = [len(plan.get(mode, [])) for plan in attribution_record_plans]
                    if any(counts):
                        print(
                            f"[stage2] {mode} streaming shards: "
                            + ", ".join(
                                f"worker {index + 1}={count}"
                                for index, count in enumerate(counts)
                            )
                        )
                if len(runtime_values) == 1:
                    compute_scalar_kl_attribution(
                        config,
                        runtime_values[0],
                        directories,
                        output,
                        record_paths_by_mode=attribution_record_plans[0],
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix=".stage2_shards_", dir=run_dir) as temporary:
                        shard_directories = [Path(temporary) / f"worker_{index}" for index in range(len(runtime_values))]
                        with ThreadPoolExecutor(max_workers=len(runtime_values)) as executor:
                            futures = [
                                executor.submit(
                                    compute_scalar_kl_attribution,
                                    config,
                                    worker_runtime,
                                    directories,
                                    shard_directories[index],
                                    shard_index=index,
                                    shard_count=len(runtime_values),
                                    record_paths_by_mode=attribution_record_plans[index],
                                )
                                for index, worker_runtime in enumerate(runtime_values)
                            ]
                            for future in futures:
                                future.result()
                        merge_attribution_shards(
                            shard_directories,
                            output,
                            primary_mode=str(config["attribution"]["primary_interpretation_mode"]),
                            methods=methods,
                        )
        expected = {"support_mass_distribution.json"} if bool(config["attribution"]["enabled"]) else set()
        for method in methods:
            prefix = "" if method == "ig" else f"{method}_"
            for base in (
                "total_response_credit.json",
                "diagnostics.json",
                "summary.json",
            ):
                expected.add(f"{prefix}{base}")
                if not bool(config["attribution"]["enabled"]):
                    write_json(
                        output / f"{prefix}{base}",
                        {"status": "disabled", "records": [], "attribution_method": method},
                    )
        prune_generated_directory(output, expected)
        return {"outputs": [output], "warnings": []}

    stage_functions[2] = (
        stage_attribution,
        (1,),
        ("attribution", "phrases", "model", "edit"),
        (),
    )

    def stage_ablations():
        output = run_dir / "ablations"
        output.mkdir(parents=True, exist_ok=True)
        ablation_summaries = load_record_summaries(run_dir / "teacher_forced")
        qualitative_keys = select_qualitative_record_keys(
            ablation_summaries,
            fraction=float(config.get("attribution", {}).get("qualitative_fraction", 0.01)),
            seed=int(config.get("run", {}).get("seed", 0)),
        )
        if bool(config["ablations"]["edit_events"]):
            runtime_values = runtimes()
            ablation_record_plans = prepare_ablation_record_shards(
                config,
                run_dir / "teacher_forced",
                shard_count=len(runtime_values),
            )
            counts = [len(paths) for paths in ablation_record_plans]
            print(
                "[stage3] streaming shards: "
                + ", ".join(
                    f"worker {index + 1}={count}"
                    for index, count in enumerate(counts)
                ),
                flush=True,
            )
            if len(runtime_values) == 1:
                event_artifact = compute_edit_event_ablations(
                    runtime_values[0],
                    None,
                    run_dir / "attribution",
                    output,
                    config,
                    record_paths=ablation_record_plans[0],
                    qualitative_record_keys=qualitative_keys,
                )
            else:
                with tempfile.TemporaryDirectory(prefix=".stage3_shards_", dir=run_dir) as temporary:
                    shard_root = Path(temporary)
                    shard_directories = [
                        shard_root / f"worker_{index}"
                        for index in range(len(runtime_values))
                    ]

                    def ablation_worker(worker_index: int):
                        return compute_edit_event_ablations(
                            runtime_values[worker_index],
                            None,
                            run_dir / "attribution",
                            shard_directories[worker_index],
                            config,
                            record_paths=ablation_record_plans[worker_index],
                            qualitative_record_keys=qualitative_keys,
                        )

                    with ThreadPoolExecutor(max_workers=len(runtime_values)) as executor:
                        futures = [
                            executor.submit(ablation_worker, index)
                            for index in range(len(runtime_values))
                        ]
                        for future in futures:
                            future.result()
                    event_artifact = merge_edit_event_ablation_shards(
                        shard_directories, output, config
                    )
        else:
            event_artifact = {"status": "disabled", "records": []}
            write_json(output / "edit_event_ablation.json", event_artifact)
        write_json(
            output / "summary.json",
            {
                "attribution_methods": event_artifact.get("attribution_methods", []),
                "primary_attribution_method": event_artifact.get("primary_attribution_method", "ig"),
                "top_events_per_method": event_artifact.get("top_events_per_method"),
                "random_events_per_method": event_artifact.get("random_events_per_method"),
                "selection_contract": event_artifact.get("selection_contract"),
                "edit_event_ablation_records": len(event_artifact.get("records", [])),
                "evaluated_events_by_method": event_artifact.get("evaluated_events_by_method", {}),
                "signed_top_events_by_method": event_artifact.get("signed_top_events_by_method", {}),
                "validation_by_method": event_artifact.get("validation_by_method", {}),
            },
        )
        prune_generated_directory(
            output,
            {"edit_event_ablation.json", "summary.json"},
        )
        return [output]

    stage_functions[3] = (
        stage_ablations,
        (2,),
        ("ablations", "phrases", "model", "edit"),
        (),
    )

    def stage_native_hypothesis_analysis():
        output = run_dir / "native_analysis"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        native_config = config.get("native_analysis", {})
        if not bool(native_config.get("enabled", True)):
            write_json(output / "summary.json", {"status": "disabled", "records": 0, "sources": 0})
            return [output]
        print(
            "[stage4] batching: "
            f"native_record_batch={int(native_config.get('native_record_batch_size', 2))} "
            f"native_condition_batch={int(native_config.get('native_condition_batch_size', 16))}",
            flush=True,
        )
        summaries = load_record_summaries(run_dir / "teacher_forced")
        selected_summaries = select_native_record_summaries(
            summaries,
            directions=native_config.get("directions", ["c_to_i", "i_to_c"]),
            one_per_source_direction=bool(native_config.get("one_record_per_source_direction", True)),
            maximum=int(native_config.get("max_records", 0)),
        )
        qualitative_keys = select_qualitative_record_keys(
            summaries,
            fraction=float(config.get("attribution", {}).get("qualitative_fraction", 0.01)),
            seed=int(config.get("run", {}).get("seed", 0)),
        )

        teacher_record_dir = run_dir / "teacher_forced" / "records"

        def record_paths_for_summaries(selected_group):
            paths = []
            for row in selected_group:
                path = teacher_record_dir / f"{row['record_id']}.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing teacher-forced tensor record: {path}")
                paths.append(path)
            return paths

        # Stage 4 tests the semantic hypotheses only in the untouched native
        # model via activation patching. Learned-editor semantic ablations were
        # retired; release any editor runtimes before loading native replicas.
        release_edit_runtimes()
        runtime_values = native_runtimes()
        if len(runtime_values) == 1:
            native_paths = record_paths_for_summaries(selected_summaries)
            print(f"[stage4] native-patching streaming records: {len(native_paths)}", flush=True)
            native_summary = compute_native_hypothesis_analysis(
                config, runtime_values[0], None, output,
                qualitative_record_keys=qualitative_keys,
                record_paths=native_paths,
            )
        else:
            groups = partition_summaries_by_source(selected_summaries, len(runtime_values))
            with tempfile.TemporaryDirectory(prefix=".stage4_native_shards_", dir=run_dir) as temporary:
                shard_root = Path(temporary)
                shard_directories = [shard_root / f"worker_{index}" for index in range(len(runtime_values))]

                native_path_groups = [record_paths_for_summaries(group) for group in groups]
                print(
                    "[stage4] native-patching streaming shards: "
                    + ", ".join(
                        f"worker {index + 1}={len(paths)}"
                        for index, paths in enumerate(native_path_groups)
                    ),
                    flush=True,
                )

                def native_worker(worker_index: int):
                    return compute_native_hypothesis_analysis_shard(
                        config,
                        runtime_values[worker_index],
                        None,
                        shard_directories[worker_index],
                        qualitative_record_keys=qualitative_keys,
                        record_paths=native_path_groups[worker_index],
                    )

                with ThreadPoolExecutor(max_workers=len(runtime_values)) as executor:
                    futures = [executor.submit(native_worker, index) for index in range(len(runtime_values))]
                    for future in futures:
                        future.result()
                native_summary = merge_native_hypothesis_shards(
                    config, shard_directories, output
                )

        # The stage-4 HTML only needs a small representative subset. Let the
        # report builder stream/rank native JSON records itself rather than
        # retaining every per-record payload in host RAM.
        build_native_hypothesis_report(config, output, None, native_summary)
        return [output]

    stage_functions[4] = (
        stage_native_hypothesis_analysis,
        (1,),
        ("native_analysis", "model", "data", "edit", "ablations"),
        (),
    )

    def stage_exhaustive_validation():
        output = run_dir / "exhaustive_validation"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        settings = config.get("exhaustive_validation", {})
        if not bool(settings.get("enabled", True)):
            summary = {"status": "disabled"}
            write_json(output / "summary.json", summary)
            return [output]

        records_directory = run_dir / "teacher_forced" / "records"
        exhaustive_paths = select_random_record_paths(
            records_directory,
            count=int(settings.get("records", 10)),
            seed=int(config["run"]["seed"]) + 600_001,
        )
        release_native_runtimes()
        runtime_values = runtimes()
        groups = partition_record_paths(exhaustive_paths, len(runtime_values))
        exhaustive_path = output / "exhaustive_token_validation.json"
        if len(runtime_values) == 1:
            exhaustive = compute_exhaustive_edit_event_validation(
                runtime_values[0],
                groups[0],
                run_dir / "attribution",
                exhaustive_path,
                config,
            )
        else:
            with tempfile.TemporaryDirectory(prefix=".stage5_exhaustive_shards_", dir=run_dir) as temporary:
                shard_paths = [Path(temporary) / f"worker_{index}.json" for index in range(len(groups))]

                def exhaustive_worker(worker_index: int):
                    return compute_exhaustive_edit_event_validation(
                        runtime_values[worker_index],
                        groups[worker_index],
                        run_dir / "attribution",
                        shard_paths[worker_index],
                        config,
                    )

                with ThreadPoolExecutor(max_workers=len(runtime_values)) as executor:
                    futures = [executor.submit(exhaustive_worker, index) for index in range(len(groups))]
                    for future in futures:
                        future.result()
                exhaustive = merge_exhaustive_validation_shards(
                    shard_paths, exhaustive_path, config
                )

        render_exhaustive_validation_html(exhaustive, output / "index.html")
        write_json(
            output / "summary.json",
            {
                "records_exhaustively_ablated": int(exhaustive.get("records_selected", 0)),
                "events_exhaustively_ablated": int(exhaustive.get("events_tested", 0)),
            },
        )
        return [output]

    stage_functions[5] = (
        stage_exhaustive_validation,
        (1, 2),
        ("exhaustive_validation", "ablations", "attribution", "model", "edit"),
        (),
    )

    def stage_sentence_browser():
        output = run_dir / "sentence_browser"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        settings = config.get("sentence_browser", {})
        if not bool(settings.get("enabled", True)):
            write_json(output / "summary.json", {"status": "disabled"})
            return [output]
        sentences = build_keyword_sentence_browser(
            run_dir / "teacher_forced" / "records",
            output / "keyword_sentences.json",
            count=int(settings.get("records", 500)),
            keywords=[str(value) for value in settings.get("keywords", [])],
            seed=int(config["run"]["seed"]) + 700_002,
        )
        render_sentence_browser_html(sentences, output / "index.html")
        write_json(
            output / "summary.json",
            {
                "records_sampled": int(sentences.get("records_sampled", 0)),
                "sentences_matched": int(sentences.get("sentences_matched", 0)),
                "keywords": list(sentences.get("keywords", [])),
            },
        )
        return [output]

    stage_functions[6] = (
        stage_sentence_browser,
        (1,),
        ("sentence_browser", "data"),
        (),
    )

    def stage_report():
        if bool(config["report"]["generate_html"]):
            build_report(config, run_dir)
        else:
            write_json(run_dir / "report" / "report_manifest.json", {"status": "disabled"})
        return [run_dir / "report"]

    stage_functions[7] = (
        stage_report,
        # The final report consumes native stage-4 outputs when available.
        # Keeping that as a formal dependency makes the numeric stage order
        # match the scientific execution order: native analysis, then report.
        (1, 2, 3, 4, 5, 6),
        ("report", "data", "attribution", "exhaustive_validation", "sentence_browser"),
        report_external_inputs(config),
    )

    requested = set(selected)
    dependency_closure = set(selected)
    pending = list(selected)
    while pending:
        index = pending.pop()
        for dependency in stage_functions[index][1]:
            if dependency not in dependency_closure:
                dependency_closure.add(dependency)
                pending.append(dependency)
    # Execute the dependency closure in topological order. The numbered stages
    # follow that order directly: focused analysis immediately
    # precedes final report assembly.
    selected = topological_stage_order(
        dependency_closure,
        {index: values[1] for index, values in stage_functions.items()},
    )

    for index in selected:
        function, dependencies, sections, external = stage_functions[index]
        runner.run(
            index,
            function,
            dependencies=dependencies,
            config_sections=sections,
            external_inputs=external,
            force=bool(args.force and index in requested),
        )

    print(
        json.dumps(
            {
                "counterfactual_analysis": "complete",
                "run_dir": str(run_dir.resolve()),
                "requested_stages": sorted(requested),
                "checked_stages": selected,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
