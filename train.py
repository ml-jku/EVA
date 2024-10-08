import os
import torch
import math
import torch.distributed
from transformers import AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
import wandb

from src.training_args import TrainingArguments
from src.utils import get_adapter_model, match_module_name, get_wandb_run_name
from src.trainer import Trainer
from src.prepare_data import load_data


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Huggingface model name"})
    model_path: str = field(default=None, metadata={"help": "Path to the model."})
    lora_dim: int = field(default=16, metadata={"help": "The dimension of the adapter."})
    lora_alpha: int = field(default=1, metadata={"help": "The alpha value of the adapter."})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout rate of the adapter."})
    adapter_type: str = field(default="lora", metadata={"help": "One of lora, adalora, dora"})
    lora_init: str = field(default="true", metadata={"help": "true, eva, gaussian, olora, pissa, pissa_niter_[number of iters], loftq"})
    redistribute: bool = field(default=False, metadata={"help": "Wether to redistribute the adapter weights."})
    target_modules: list[str] = field(default=None, metadata={"help": "The target modules for the adapter."})
    ignore_modules: list[str] = field(default=None, metadata={"help": "The modules to ignore."})
    n_components_for_init: int = field(default=None,
        metadata={"help": "The number of components to initialize the adapter with. Remaining components will be initialized randomly"}
    )
    model_max_length: int = field(default=None, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    svd_filepath: str = field(default=None, metadata={"help": "Path to the SVD checkpoint file"})


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Path to the training data."})
    dataset_path: str = field(default=None, metadata={"help": "Optional local path to the training data. Can be the same as dataset_name."})
    filter_long_context_samples: bool = field(default=False, metadata={"help": "Filter out samples with context length > model_max_length."})

        

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if model_args.model_path is None:
        model_args.model_path = model_args.model_name
    # get around not being able to use multiple types in the same dataclass field
    if model_args.lora_init.lower() == "true":
        model_args.lora_init = True
    # setting this to false to avoid issues with columns that are needed by the data collator
    training_args.remove_unused_columns = False

    if torch.distributed.get_rank() == 0:
        print(model_args)
        print(data_args)
        print(training_args)

    torch.manual_seed(training_args.seed)

    if 'gemma-2-9b' in model_args.model_path:
        kwargs = {'attn_implementation': 'eager'}
    else:
        kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(model_args.model_path, **kwargs)

    if model_args.target_modules is None:
        if 'gemma-2-9b' in model_args.model_path and data_args.dataset_name == "m-a-p/Code-Feedback":
            model_args.target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not 'lm_head' in n]
        else:
            model_args.target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        
    datasets, data_collator = load_data(
        data_args.dataset_name,
        data_args.dataset_path,
        model_args.model_path,
        data_args.filter_long_context_samples,
        model_args.model_max_length,
        interleave=False,
        seed=training_args.seed
    )

    model.cuda()

    svd_dict = None
    if model_args.redistribute or model_args.lora_init == "eva":
        assert os.path.isfile(model_args.svd_filepath), f"SVD checkpoint file {model_args.svd_filepath} does not exist"
        svd_dict = torch.load(model_args.svd_filepath, map_location="cpu")
        all_target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                add_module = True
                if model_args.target_modules is not None:
                    add_module = any([match_module_name(name, t) for t in model_args.target_modules])
                if model_args.ignore_modules is not None:
                    add_module = not any([match_module_name(name, i) for i in model_args.ignore_modules])
                if add_module:
                    all_target_modules.append(name)
        assert all(k in svd_dict for k in all_target_modules), "Missing keys in svd_dict"

    total_steps = math.ceil(len(datasets["train"]) / training_args.per_device_train_batch_size) * training_args.num_train_epochs
    model = get_adapter_model(model, model_args, svd_dict=svd_dict, total_steps=total_steps)

    # run name for wandb
    run_name = get_wandb_run_name(
        model_args.model_name,
        data_args.dataset_name,
        model_args.adapter_type,
        model_args.lora_dim,
        model_args.lora_init,
        model_args.redistribute,
        model_args.svd_filepath,
        model_args.n_components_for_init
    )

    # save initial adapter state (needed for pissa and olora)
    if torch.distributed.get_rank() == 0:
        model.save_pretrained(training_args.output_dir)
        os.rename(
            f"{training_args.output_dir}/adapter_model.safetensors",
            f"{training_args.output_dir}/initial_adapter_model.safetensors"
        )

        wandb_config = {}
        wandb_config.update({f"model_args.{k}": str(v) for k, v in model_args.__dict__.items()})
        wandb_config.update({f"data_args.{k}": str(v) for k, v in data_args.__dict__.items()})
        wandb_config.update({f"training_args.{k}": str(v) for k, v in training_args.__dict__.items()})
        wandb.init(
            name=run_name,
            config=wandb_config
        )

    setattr(training_args, "run_name", run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"] if "validation" in datasets else None,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()

    if torch.distributed.get_rank() == 0:
        trainer.model.save_pretrained(training_args.output_dir)
        wandb.finish()


if __name__ == "__main__":
    main()
