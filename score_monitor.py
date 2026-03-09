# -*- coding: utf-8 -*-

import json
import logging
import os
import random
import re
import shutil
import socket
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import datasets
import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from utils import init

accelerator = Accelerator()
log = logging.getLogger(__name__)

if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x


def log_color(content: Any, title: str = "") -> None:
    try:
        console = Console()
        console.print(Panel(content, title=title, border_style="cyan", title_align="left"))

        string_io = StringIO()
        plain_console = Console(file=string_io, highlight=False)
        plain_console.print(Panel(content, title=title, border_style="none", title_align="left"))
        log.info("\n" + string_io.getvalue())
    except Exception as exc:
        log.info("Error logging content: %s", exc)


def flatten_dict(value: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for key, child in value.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(child, dict):
            items.extend(flatten_dict(child, new_key, sep=sep).items())
        else:
            items.append((new_key, child))
    return dict(items)


def resolve_dtype(dtype_name: str):
    mapping = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={dtype_name}")
    return mapping[dtype_name]


def load_prompt(prompt_path: str, allow_prompt_scaffold: bool) -> str:
    prompt_text = Path(prompt_path).read_text()
    if "PROMPT_SCAFFOLD_MARKER" in prompt_text and not allow_prompt_scaffold:
        raise ValueError(
            "Prompt scaffold is still active. Replace prompts/monitorability_2510_23966_prompt.txt "
            "with the exact Appendix C prompt or set allow_prompt_scaffold=true for pipeline testing only."
        )
    return prompt_text


def extract_trace_final_answer(trace_text: str) -> str:
    if not trace_text:
        return ""

    answer_force_marker = "**Final Answer**"
    if answer_force_marker in trace_text:
        return trace_text.split(answer_force_marker)[-1].strip()

    boxed_matches = re.findall(r"\\boxed\{.*?\}", trace_text, flags=re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()

    lines = [line.strip() for line in trace_text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_monitor_response(raw_response: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "legibility_score": None,
        "coverage_score": None,
        "parse_success": False,
        "parse_error": None,
        "judge_explanation": None,
    }

    candidate_blocks = []
    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL)
    candidate_blocks.extend(fenced_matches)

    brace_matches = re.findall(r"(\{.*\})", raw_response, flags=re.DOTALL)
    candidate_blocks.extend(brace_matches)

    for candidate in candidate_blocks:
        try:
            payload = json.loads(candidate)
            legibility_score = payload.get("legibility_score")
            coverage_score = payload.get("coverage_score")
            if isinstance(legibility_score, int) and isinstance(coverage_score, int):
                parsed["legibility_score"] = legibility_score
                parsed["coverage_score"] = coverage_score
                parsed["judge_explanation"] = payload.get("justification", payload.get("explanation"))
                parsed["parse_success"] = all(0 <= score <= 4 for score in [legibility_score, coverage_score])
                if not parsed["parse_success"]:
                    parsed["parse_error"] = "Scores must be integers in [0, 4]"
                return parsed
        except json.JSONDecodeError:
            continue

    legibility_match = re.search(r"legibility(?:_score)?[^0-4-]*([0-4])", raw_response, flags=re.IGNORECASE)
    coverage_match = re.search(r"coverage(?:_score)?[^0-4-]*([0-4])", raw_response, flags=re.IGNORECASE)
    if legibility_match and coverage_match:
        parsed["legibility_score"] = int(legibility_match.group(1))
        parsed["coverage_score"] = int(coverage_match.group(1))
        parsed["parse_success"] = True
        return parsed

    parsed["parse_error"] = "Unable to parse legibility_score and coverage_score from judge response"
    return parsed


def build_monitor_prompt(template: str, example: dict[str, Any]) -> str:
    return template.format(
        question=example["problem"],
        explanation=example["monitor_target_text"],
        answer=example.get("trace_final_answer", ""),
        problem=example["problem"],
        ground_truth_solution=example.get("solution", ""),
        trace_text=example["monitor_target_text"],
        trace_column=example["monitor_target_column"],
        trace_final_answer=example.get("trace_final_answer", ""),
    )


def load_trace_metadata(trace_path: str) -> dict[str, Any]:
    yaml_path = Path(trace_path + ".yaml")
    if not yaml_path.exists():
        return {}
    return OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)


def add_constant_column(dataset: datasets.Dataset, name: str, value: Any) -> datasets.Dataset:
    return dataset.add_column(name, [value] * len(dataset))


def explode_trace_columns(trace_dataset: datasets.Dataset, trace_path: str, trace_columns: list[str], only_correct: bool) -> list[datasets.Dataset]:
    metadata = load_trace_metadata(trace_path)
    trace_name = metadata.get("trace_name", Path(trace_path).name)
    exploded = []

    trace_dataset = add_constant_column(trace_dataset, "source_trace_path", trace_path)
    trace_dataset = add_constant_column(trace_dataset, "source_trace_name", trace_name)
    trace_dataset = add_constant_column(trace_dataset, "source_data_split", metadata.get("data_split"))
    trace_dataset = add_constant_column(trace_dataset, "source_tau", metadata.get("tau"))
    trace_dataset = add_constant_column(trace_dataset, "source_lam", metadata.get("lam"))
    trace_dataset = add_constant_column(trace_dataset, "source_eps", metadata.get("eps"))
    trace_dataset = add_constant_column(trace_dataset, "source_teacher", metadata.get("teacher"))
    trace_dataset = trace_dataset.add_column("source_row_index", list(range(len(trace_dataset))))

    for trace_column in trace_columns:
        if trace_column not in trace_dataset.column_names:
            continue

        correct_column = "is_raw_correct" if trace_column == "trace" else "is_af_correct"
        view = trace_dataset
        if only_correct and correct_column in view.column_names:
            view = view.filter(lambda example: bool(example[correct_column]), desc=f"Filtering correct rows for {trace_name}:{trace_column}")

        if len(view) == 0:
            continue

        view = add_constant_column(view, "monitor_target_column", trace_column)
        view = add_constant_column(view, "monitor_target_is_correct_column", correct_column)
        view = view.add_column("monitor_target_is_correct", list(view[correct_column]) if correct_column in view.column_names else [None] * len(view))
        view = view.add_column("monitor_target_text", list(view[trace_column]))
        view = view.add_column("trace_final_answer", [extract_trace_final_answer(text) for text in view[trace_column]])
        view = view.add_column("ground_truth_solution", list(view["solution"]) if "solution" in view.column_names else [""] * len(view))
        exploded.append(view)

    return exploded


class JudgeBackend:
    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError


class TransformersJudgeBackend(JudgeBackend):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.monitor.tokenizer_name or cfg.monitor.model_name,
            trust_remote_code=cfg.monitor.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if accelerator.num_processes > 1 and cfg.monitor.device_map is not None:
            raise ValueError("device_map cannot be used with multi-process Accelerate launches")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.monitor.trust_remote_code,
            "torch_dtype": resolve_dtype(cfg.monitor.torch_dtype),
        }
        if cfg.monitor.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.monitor.attn_implementation
        if cfg.monitor.device_map is not None:
            model_kwargs["device_map"] = cfg.monitor.device_map

        self.model = AutoModelForCausalLM.from_pretrained(cfg.monitor.model_name, **model_kwargs)
        if cfg.monitor.device_map is None:
            self.model = self.model.to(accelerator.device)
        self.model.eval()

    def _input_device(self):
        if self.cfg.monitor.device_map is not None:
            return next(self.model.parameters()).device
        return accelerator.device

    def generate(self, prompts: list[str]) -> list[str]:
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.monitor.max_prompt_length,
        )
        tokenized = {key: value.to(self._input_device()) for key, value in tokenized.items()}
        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.cfg.monitor.max_new_tokens,
                do_sample=self.cfg.monitor.do_sample,
                temperature=None if self.cfg.monitor.temperature == 0 else self.cfg.monitor.temperature,
                top_p=self.cfg.monitor.top_p,
                repetition_penalty=self.cfg.monitor.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = []
        for idx, output_ids in enumerate(outputs):
            generated_ids = output_ids[prompt_lengths[idx]:]
            decoded.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        return decoded


class VllmJudgeBackend(JudgeBackend):
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError(
            "vLLM backend is not implemented in this first pass. "
            "Use monitor.backend=transformers for now and add vLLM when you are ready to optimize 70B/120B inference."
        )


class ApiJudgeBackend(JudgeBackend):
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError(
            "API judge backend is not implemented yet. The script structure preserves a backend seam for frontier-model scoring later."
        )


def build_backend(cfg: DictConfig) -> JudgeBackend:
    backend_name = cfg.monitor.backend
    if backend_name == "transformers":
        return TransformersJudgeBackend(cfg)
    if backend_name == "vllm":
        return VllmJudgeBackend(cfg)
    if backend_name == "api":
        return ApiJudgeBackend(cfg)
    raise ValueError(f"Unknown monitor backend: {backend_name}")


def compute_summary(scored_dataset: datasets.Dataset) -> dict[str, Any]:
    df = scored_dataset.to_pandas()
    parse_success_rate = float(df["parse_success"].mean()) if len(df) else 0.0
    successful_df = df[df["parse_success"]]

    summary: dict[str, Any] = {
        "count": int(len(df)),
        "parse_success_rate": parse_success_rate,
    }
    if len(successful_df) == 0:
        return summary

    summary["legibility_score_mean"] = float(successful_df["legibility_score"].mean())
    summary["coverage_score_mean"] = float(successful_df["coverage_score"].mean())
    summary["legibility_percent_mean"] = float(successful_df["legibility_score"].mean() / 4.0 * 100.0)
    summary["coverage_percent_mean"] = float(successful_df["coverage_score"].mean() / 4.0 * 100.0)

    grouped = (
        successful_df
        .groupby(["source_trace_name", "monitor_target_column"], dropna=False)
        [["legibility_score", "coverage_score"]]
        .mean()
        .reset_index()
    )
    summary["by_trace_and_column"] = grouped.to_dict(orient="records")
    return summary


def iter_prompt_batches(dataset: datasets.Dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield dataset.select(range(start, end))


@hydra.main(config_path=".", config_name="monitor_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.input_trace_paths:
        raise ValueError("input_trace_paths must contain at least one saved trace dataset path")

    init(os.getenv("USER"), cfg.seed, "babel" in socket.gethostname())
    prompt_template = load_prompt(cfg.monitor.prompt_path, cfg.allow_prompt_scaffold)
    Path(cfg.run_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.monitor_registry).parent.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        config_content = Syntax(OmegaConf.to_yaml(cfg, resolve=True), "yaml", theme="monokai")
        log_color(config_content, title="Monitor Config")

    if accelerator.is_main_process:
        exploded_datasets = []
        for trace_path in cfg.input_trace_paths:
            trace_dataset = datasets.load_from_disk(trace_path)
            exploded_datasets.extend(explode_trace_columns(trace_dataset, trace_path, list(cfg.trace_columns), cfg.only_correct))

        if not exploded_datasets:
            raise ValueError("No rows matched the requested trace_columns and correctness filter")

        combined_dataset = datasets.concatenate_datasets(exploded_datasets)
        combined_dataset = combined_dataset.map(
            lambda example: {"monitor_prompt": build_monitor_prompt(prompt_template, example)},
            desc="Rendering monitor prompts",
        )
        temp_input_dir = Path(tempfile.mkdtemp(prefix="monitor_input_"))
        combined_dataset.save_to_disk(temp_input_dir)
    else:
        temp_input_dir = None

    accelerator.wait_for_everyone()
    all_temp_dirs = gather_object([str(temp_input_dir) if temp_input_dir is not None else None])
    input_dir = next(Path(item) for item in all_temp_dirs if item)
    combined_dataset = datasets.load_from_disk(str(input_dir))

    backend = build_backend(cfg)
    dataset_shard = combined_dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

    raw_responses = []
    parsed_rows = []
    total_batches = (len(dataset_shard) + cfg.monitor.batch_size - 1) // cfg.monitor.batch_size
    for batch in tqdm(iter_prompt_batches(dataset_shard, cfg.monitor.batch_size), total=total_batches, desc="Scoring monitor prompts"):
        responses = backend.generate(batch["monitor_prompt"])
        raw_responses.extend(responses)
        parsed_rows.extend(parse_monitor_response(response) for response in responses)

    dataset_shard = dataset_shard.add_column("monitor_raw_response", raw_responses)
    dataset_shard = dataset_shard.add_column("legibility_score", [row["legibility_score"] for row in parsed_rows])
    dataset_shard = dataset_shard.add_column("coverage_score", [row["coverage_score"] for row in parsed_rows])
    dataset_shard = dataset_shard.add_column("parse_success", [row["parse_success"] for row in parsed_rows])
    dataset_shard = dataset_shard.add_column("parse_error", [row["parse_error"] for row in parsed_rows])
    dataset_shard = dataset_shard.add_column("judge_explanation", [row["judge_explanation"] for row in parsed_rows])
    dataset_shard = dataset_shard.add_column("judge_backend", [cfg.monitor.backend] * len(dataset_shard))
    dataset_shard = dataset_shard.add_column("judge_model", [cfg.monitor.model_name] * len(dataset_shard))

    tmp_dir = Path(tempfile.mkdtemp(prefix="monitor_scored_"))
    shard_path = tmp_dir / f"shard_rank_{accelerator.process_index:05d}"
    dataset_shard.save_to_disk(shard_path)

    accelerator.wait_for_everyone()
    all_paths = gather_object([str(shard_path)])
    if accelerator.is_main_process:
        scored_dataset = datasets.concatenate_datasets([datasets.load_from_disk(path) for path in all_paths])
        scored_dataset.save_to_disk(cfg.run_path)
        scored_dataset.to_parquet(cfg.run_path + ".parquet")

        for path in all_paths:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(input_dir, ignore_errors=True)

        summary = compute_summary(scored_dataset)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        hydra_cfg = HydraConfig.get()
        full_cfg["hydra"] = {
            "run_dir": hydra_cfg.run.dir,
            "job_name": hydra_cfg.job.name,
            "cwd": hydra_cfg.runtime.cwd,
        }
        if cfg.save_prompt_text:
            full_cfg["monitor"]["prompt_text"] = prompt_template
        full_cfg["stats"] = summary
        full_cfg_obj = OmegaConf.create(full_cfg)

        yaml_path = cfg.run_path + ".yaml"
        with open(yaml_path, "w") as handle:
            OmegaConf.save(config=full_cfg_obj, f=handle)

        with open(cfg.monitor_registry, "a") as handle:
            handle.write(json.dumps(flatten_dict(full_cfg)) + "\n")

        example_row = scored_dataset[random.randint(0, len(scored_dataset) - 1)]
        log_color(example_row["problem"], title="Example Problem")
        log_color(example_row["monitor_target_text"], title=f"Example Trace [{example_row['monitor_target_column']}]")
        log_color(example_row["monitor_raw_response"], title="Example Judge Response")
        log_color(Syntax(OmegaConf.to_yaml(full_cfg_obj, resolve=True), "yaml", theme="monokai"), title="Final Config")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
