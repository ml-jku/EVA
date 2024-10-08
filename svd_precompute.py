import os
import torch
from transformers import AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field

from src.svd import compute_svd
from src.utils import get_svd_fileref
from train import load_data


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Huggingface model name"})
    model_path: str = field(default=None, metadata={"help": "Path to the model."})
    svd_path: str = field(default=None, metadata={"help": "Path to the SVD model."})
    target_modules: list[str] = field(default=None, metadata={"help": "The target modules for the adapter."})
    ignore_modules: list[str] = field(default=None, metadata={"help": "The modules to ignore."})

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Path to the training data."})
    dataset_path: str = field(default=None, metadata={"help": "Optional local path to the training data. Can be the same as dataset_name."})
    filter_long_context_samples: bool = field(default=False, metadata={"help": "Filter out samples with context length > model_max_length."})

@dataclass
class TrainingArguments:
    rank: int = field(default=None)
    rho: int = field(default=2)
    early_stop_sim_thresh: float = field(default=0.99)
    early_stop_redist_metric: str = field(default="ratio")
    scale_by_singular_values: bool = field(default=False)
    batch_size: int = field(default=4)
    model_max_length: int = field(default=None, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    log_convergence_stats: bool = field(default=False, metadata={"help": "Log convergence stats."})
    whiten: bool = field(default=False, metadata={"help": "Whiten the data before performing SVD."})
    use_label_mask: bool = field(default=False, metadata={"help": "Use label mask for the SVD computation."})
    min_batches: int = field(default=1, metadata={"help": "Minimum number of batches to use for SVD computation."})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if model_args.model_path is None:
        model_args.model_path = model_args.model_name

    torch.manual_seed(training_args.seed)

    if 'gemma-2-9b' in model_args.model_path:
        kwargs = {'attn_implementation': 'eager'}
    else:
        kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(model_args.model_path, **kwargs)

    if model_args.target_modules is None:
        model_args.target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

    datasets, data_collator = load_data(
        data_args.dataset_name,
        data_args.dataset_path,
        model_args.model_path,
        data_args.filter_long_context_samples,
        training_args.model_max_length,
        interleave=(data_args.dataset_name == "qa_datasets"),
        seed=training_args.seed
    )

    fileref = get_svd_fileref(
        model_args.model_name,
        data_args.dataset_name,
        training_args.batch_size,
        training_args.seed,
        training_args.rank,
        training_args.rho,
        training_args.early_stop_sim_thresh,
        training_args.early_stop_redist_metric,
        training_args.scale_by_singular_values,
        training_args.whiten,
        training_args.use_label_mask,
        training_args.model_max_length
    )
    svd_filepath = os.path.join(model_args.svd_path, fileref + "_svd.bin")
    if not os.path.exists(svd_filepath):
        os.makedirs(model_args.svd_path, exist_ok=True)
        svd_data_loader = torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=training_args.batch_size,
            collate_fn=data_collator,
            shuffle=(data_args.dataset_name != "qa_datasets")
        )
        model.cuda()
        svd_dict, has_converged_stats = compute_svd(
            model=model,
            data_loader=svd_data_loader,
            rank=training_args.rank,
            rho=training_args.rho,
            early_stop_sim_thresh=training_args.early_stop_sim_thresh,
            early_stop_redist_metric=training_args.early_stop_redist_metric,
            scale_by_singular_values=training_args.scale_by_singular_values,
            whiten=training_args.whiten,
            target_modules=model_args.target_modules,
            ignore_modules=model_args.ignore_modules,
            use_label_mask=training_args.use_label_mask,
            min_batches=training_args.min_batches,
            log_convergence_stats=training_args.log_convergence_stats
        )
        torch.save(svd_dict, svd_filepath)
        if has_converged_stats is not None:
            torch.save(has_converged_stats, svd_filepath.replace("svd.bin", "convergence_stats.bin"))
    else:
        print(f"{svd_filepath} already exists")
 

if __name__ == "__main__":
    main()
