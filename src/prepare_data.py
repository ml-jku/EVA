import torch
import re
from functools import partial
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset, concatenate_datasets, interleave_datasets
from promptsource_custom.templates import DatasetTemplates


def _tokenize_fn(prompts, completions, tokenizer):
    prompt_tokens = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    input_tokens = tokenizer([x+y for x, y in zip(prompts, completions)], add_special_tokens=False)["input_ids"]
    input_tokens = [[tokenizer.bos_token_id]+x+[tokenizer.eos_token_id] for x in input_tokens]
    prompt_length = [len(x)+1 for x in prompt_tokens] # +1 for the bos token
    input_length = [len(x) for x in input_tokens]
    return {"input_ids": input_tokens, "prompt_length": prompt_length, "input_length": input_length}


class _TokenizerPromptSource:

    def __init__(self, tokenizer_path, space_after_prompt=True):
 
        self.dataset_templates = DatasetTemplates
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.space_after_prompt = space_after_prompt

    def __call__(self, examples):
        examples = [dict(zip(examples.keys(), e)) for e in zip(*examples.values())]
        prompts, completions = zip(*[self.prompt.apply(e) for e in examples])
        if self.space_after_prompt:
            prompts = [p + " " for p in prompts]
        return _tokenize_fn(prompts, completions, self.tokenizer)


class TokenizerMetaMath:

    PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response: "
    )
    PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Input:\n{input}\n\n### Response: "        
    )

    def format_prompt(self, query):
        query = query.split("\n", 1)
        if len(query) == 1 or query[1].strip("\n") == "":
            return self.PROMPT_NO_INPUT.format(query=query[0])
        else:
            return self.PROMPT.format(query=query[0], input=query[1])

    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, examples):
        prompts = [self.format_prompt(text) for text in examples["query"]]
        completions = examples["response"]
        return _tokenize_fn(prompts, completions, self.tokenizer)
    

class TokenizerCodeFeedback:

    PROMPT = (
        "Instruction:\nGiven a multi-turn dialogue related to a coding task, your role is to generate the assistant's next response."
        "\n\nDialogue:\n"
    )
    CHAT_TEMPLATE_PATH = "chat_template.jinja"

    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenized_prompt = self.tokenizer(self.PROMPT, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        self.tokenized_prompt_len = len(self.tokenized_prompt)

        chat_template = open(self.CHAT_TEMPLATE_PATH).read()
        self.tokenizer.chat_template = chat_template

    def __call__(self, examples):
        chats = self.tokenizer.apply_chat_template(
            examples["messages"],
            add_generation_prompt=False,
            return_dict=True,
            tokenize=True,
            return_assistant_tokens_mask=True,
            tokenizer_kwargs={'return_attention_mask': False, 'return_length': True}
        )
        input_ids = [self.tokenized_prompt + x for x in chats["input_ids"]]
        assistant_mask = [[0] * self.tokenized_prompt_len + x for x in chats["assistant_masks"]]
        input_length = [x + self.tokenized_prompt_len for x in chats["length"]]
        return {"input_ids": input_ids, "assistant_mask": assistant_mask, 'input_length': input_length}
    

# to circumvent issue
# https://github.com/huggingface/transformers/issues/33091
class TokenizerCodeFeedbackHacky:

    PROMPT = (
        "Instruction:\nGiven a multi-turn dialogue related to a coding task, your role is to generate the assistant's next response."
        "\n\nDialogue:\n"
    )
    CHAT_TEMPLATE_PATH = "chat_template.jinja"

    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenized_prompt = self.tokenizer(self.PROMPT, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        self.tokenized_prompt_len = len(self.tokenized_prompt)

        chat_template = open(self.CHAT_TEMPLATE_PATH).read()
        self.tokenizer.chat_template = chat_template
        self.chat_template_header = "### {role}:\n" #.format(role="assistant")

    def __call__(self, examples):
        chats = self.tokenizer.apply_chat_template(
            examples["messages"],
            add_generation_prompt=False,
            return_dict=False,
            tokenize=False
        )
        chats_tokenized = self.tokenizer(chats, add_special_tokens=False, return_attention_mask=False, return_length=True, return_offsets_mapping=True)
        assistant_mask = []
        for i in range(len(chats)):
            s, _ = zip(*chats_tokenized[i].offsets)
            s = torch.tensor(s)
            assistant_starts = [x.end()+1 for x in re.finditer(self.chat_template_header.format(role="assistant"), chats[i])]
            assistant_ends = [x.start()-1 for x in re.finditer(self.chat_template_header.format(role="user"), chats[i])]
            assistant_ends = assistant_ends[1:] + [len(chats[i])]
            assistant_start_ids, assistant_end_ids = [], []
            for start, end in zip(assistant_starts, assistant_ends):
                assistant_start_ids.append((s > start).long().argmax().item() - 1)
                assistant_end_ids.append((s > end).long().argmax().item() - 1)
            assistant_end_ids = assistant_end_ids[:-1] + [chats_tokenized["length"][i]-1]
            mask = [0] * chats_tokenized["length"][i]
            for start_id, end_id in zip(assistant_start_ids, assistant_end_ids):
                mask[start_id:end_id] = [1] * (end_id-start_id)
            assistant_mask.append(mask)
        input_ids = [self.tokenized_prompt + x for x in chats_tokenized["input_ids"]]
        assistant_mask = [[0] * self.tokenized_prompt_len + x for x in assistant_mask]
        input_length = [x + self.tokenized_prompt_len for x in chats_tokenized["length"]]
        return {"input_ids": input_ids, "assistant_mask": assistant_mask, 'input_length': input_length}


class TokenizerWinogrande(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("winogrande", "winogrande_xl")["multiple_choice_simple"]
    

class TokenizerHellaswag(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("hellaswag")["multiple_choice_simple"]


class TokenizerArcChallenge(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("ai2_arc", "ARC-Challenge")["multiple_choice_simple"]


class TokenizerArcEasy(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("ai2_arc", "ARC-Easy")["multiple_choice_simple"]


class TokenizerPIQA(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("piqa")["multiple_choice_simple"]

class TokenizerSIQA(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("social_i_qa")["multiple_choice_simple"]

class TokenizerOpenBookQA(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("openbookqa", "main")["multiple_choice_simple"]


class TokenizerBoolQ(_TokenizerPromptSource):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.prompt = self.dataset_templates("super_glue", "boolq")["multiple_choice_simple"]



class DataCollator:
    def __init__(self, eos_token_id, max_length = None):
        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def __call__(self, batch):
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        input_lengths = torch.stack(batch["input_length"])
        prompt_lengths = torch.stack(batch["prompt_length"])
        input_ids = torch.nn.utils.rnn.pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.eos_token_id)
        col_indices = torch.arange(input_ids.size(1)).unsqueeze(0)
        attention_mask = col_indices < input_lengths.unsqueeze(1)
        label_mask = torch.logical_or(col_indices < prompt_lengths.unsqueeze(1), ~attention_mask)
        labels = input_ids.masked_fill(label_mask, -100)
        if self.max_length is not None:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    

class DataCollatorMultiTurn:

    def __init__(self, eos_token_id, max_length = None):
        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def __call__(self, batch):
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        input_lengths = torch.stack(batch["input_length"])
        input_ids = torch.nn.utils.rnn.pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.eos_token_id)
        assistant_mask = torch.nn.utils.rnn.pad_sequence(batch["assistant_mask"], batch_first=True, padding_value=0).bool()
        col_indices = torch.arange(input_ids.size(1)).unsqueeze(0)
        attention_mask = col_indices < input_lengths.unsqueeze(1)
        labels = input_ids.masked_fill(~assistant_mask, -100)
        if self.max_length is not None:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    


LOAD_DATASET_KWARGS = {
    "meta-math/MetaMathQA": {"path": "meta-math/MetaMathQA"},
    "m-a-p/Code-Feedback": {"path": "m-a-p/Code-Feedback"},
    "Rowan/hellaswag": {"path": "Rowan/hellaswag"},
    "allenai/winogrande": {"path": "allenai/winogrande", "name": "winogrande_xl"},
    "allenai/ai2_arc_challenge": {"path": "allenai/ai2_arc", "name": "ARC-Challenge"},
    "allenai/ai2_arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy"},
    "ybisk/piqa": {"path": "ybisk/piqa"},
    "allenai/social_i_qa": {"path": "allenai/social_i_qa"},
    "allenai/openbookqa": {"path": "allenai/openbookqa", "name": "main"},
    "boolq": {"path": "aps/super_glue", "name": "boolq"}
}


TOKENIZE_MAP = {
    "meta-math/MetaMathQA": TokenizerMetaMath,
    "m-a-p/Code-Feedback": TokenizerCodeFeedback,
    "Rowan/hellaswag": TokenizerHellaswag,
    "allenai/winogrande": TokenizerWinogrande,
    "allenai/ai2_arc_challenge": TokenizerArcChallenge,
    "allenai/ai2_arc_easy": TokenizerArcEasy,
    "ybisk/piqa": TokenizerPIQA,
    "allenai/social_i_qa": TokenizerSIQA,
    "allenai/openbookqa": TokenizerOpenBookQA,
    "boolq": TokenizerBoolQ
}


COLLATOR_MAP = {
    "meta-math/MetaMathQA": DataCollator,
    "m-a-p/Code-Feedback": DataCollatorMultiTurn,
    "Rowan/hellaswag": DataCollator,
    "allenai/winogrande": DataCollator,
    "allenai/ai2_arc_challenge": DataCollator,
    "allenai/ai2_arc_easy": DataCollator,
    "ybisk/piqa": DataCollator,
    "allenai/social_i_qa": DataCollator,
    "allenai/openbookqa": DataCollator,
    "boolq": DataCollator
}


QA_DATASETS = [
    "Rowan/hellaswag",
    "allenai/winogrande",
    "allenai/ai2_arc_challenge",
    "allenai/ai2_arc_easy",
    "ybisk/piqa",
    "allenai/social_i_qa",
    "allenai/openbookqa",
    "boolq"
]


def _load_data(
    dataset_name,
    dataset_path,
    model_path,
    filter_long_context_samples = False,
    model_max_length = None
):
    tokenizer_cls = TOKENIZE_MAP[dataset_name]
    if dataset_name == "m-a-p/Code-Feedback" and "meta-llama-3" in model_path.lower():
        tokenizer_cls = TokenizerCodeFeedbackHacky
    tokenizer_wrapper = tokenizer_cls(tokenizer_path=model_path)

    load_dataset_kwargs = LOAD_DATASET_KWARGS[dataset_name]
    if dataset_path is not None:
        load_dataset_kwargs["path"] = dataset_path

    datasets = load_dataset(**load_dataset_kwargs, trust_remote_code=True)
    datasets = datasets.map(
        tokenizer_wrapper,
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    datasets.set_format(type="torch")
    
    if filter_long_context_samples:
        datasets = datasets.filter(lambda example: example["input_length"] <= model_max_length)

    data_collator = COLLATOR_MAP[dataset_name](
        tokenizer_wrapper.tokenizer.eos_token_id,
        model_max_length
    )

    return datasets, data_collator


def load_data(
    dataset_name,
    dataset_path,
    model_path,
    filter_long_context_samples = False,
    model_max_length = None,
    interleave = True,
    seed = 0
):
    
    if interleave:
        merge_fn = partial(interleave_datasets, seed=seed)
    else:
        merge_fn = partial(concatenate_datasets, axis=0)
    
    if dataset_name == "qa_datasets":
        datasets = []
        for dataset_name in QA_DATASETS:
            ds, data_collator = _load_data(
                dataset_name=dataset_name,
                dataset_path=None,
                model_path=model_path,
                filter_long_context_samples=filter_long_context_samples,
                model_max_length=model_max_length
            )
            datasets.append(ds)
        all_splits = set([n for ds in datasets for n in ds.keys()])
        datasets = DatasetDict({split: merge_fn([ds[split] for ds in datasets if split in ds]) for split in all_splits})
        if not interleave_datasets:
            datasets = datasets.shuffle(seed=seed)
    else:         
        datasets, data_collator = _load_data(
            dataset_name,
            dataset_path,
            model_path,
            filter_long_context_samples,
            model_max_length
        )
    return datasets, data_collator