import os
import re
import torch
import functools
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AdaLoraConfig, PeftConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from typing import Optional


def get_eva_state_dict(
        svd_dict: dict,
        target_modules: list,
        rank: int,
        redistribute: bool = False,
        exp_vars: Optional[dict] = None,
        redist_metric: str = "ratio",
        rho: int = 2
    ):
    target_keys = list(filter(lambda k: any([match_module_name(k, t) for t in target_modules]), svd_dict.keys()))
    svd_dict = {k: v for k, v in svd_dict.items() if k in target_keys}
    available_components = svd_dict[next(iter(svd_dict))].size(0)
    if redistribute:
        assert redist_metric in ["ratio", "raw", "sum", "max"], "redist metric must be either of raw, ratio, sum, max"
        assert rank * rho <= available_components, "rank * rho must be less than the number of available components"
        assert rho >= 1, "rho must be greater than or equal to 1"
        exp_vars = {k: v[redist_metric] for k, v in exp_vars.items() if k in target_keys}
        n_components = rank * rho
        rank_budget = rank * len(svd_dict)
        exp_vars_list = [(k, v) for k, values in exp_vars.items() for v in values[:n_components]]
        exp_vars_list.sort(key=lambda x: x[1])
        counts = dict(Counter([x[0] for x in exp_vars_list[-rank_budget:]]))
        # print number of assigned ranks
        print("rank counts:\n" + "\n".join([f"rank {k}: {v}" for k,v in Counter(counts.values()).items()]))
    else:
        assert rank <= available_components, "rank must be less than the number of available components"
        counts = dict(zip(svd_dict.keys(), [rank] * len(svd_dict)))
    return {k: svd_dict[k][:v] for k, v in counts.items()}


def get_adapter_model(
    model,
    model_args,
    svd_dict = None,
    total_steps = None
):
    if model_args.adapter_type == "lora" or model_args.adapter_type == "dora":
        init_weights = model_args.lora_init
        if model_args.lora_init == "eva":
            init_weights = True
        peft_config = LoraConfig(
            init_lora_weights=init_weights,
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_dim,
            use_dora=(model_args.adapter_type == "dora"),
        )
    elif model_args.adapter_type == "adalora":
        tfinal = int(total_steps * 0.1)
        tinit = int(tfinal * 0.25)
        peft_config = AdaLoraConfig(
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_r=model_args.lora_dim,
            init_r=round(model_args.lora_dim*1.5),
            tinit=tinit,
            tfinal=tfinal,
            deltaT=10,
            orth_reg_weight=0.1,
            total_step=total_steps
        )    
    else:
        raise NotImplementedError("Adapter type not implemented")

    if model_args.redistribute or model_args.lora_init == "eva":
        assert isinstance(peft_config, LoraConfig), "only lora adapters are supported for eva initialization"
        assert svd_dict is not None, "svd_dict must be provided for redistribution or eva initialization"
        peft_config.target_modules = list(svd_dict.keys())
        peft_config.rank_pattern = {k: v.size(0) for k, v in svd_dict.items()}
        if model_args.target_modules:
            svd_target_modules = [name for name in svd_dict.keys() if any([match_module_name(name, t) for t in model_args.target_modules])]
            peft_config.target_modules = [name for name in peft_config.target_modules if name in svd_target_modules]
            peft_config.rank_pattern = {k: v for k, v in peft_config.rank_pattern.items() if k in peft_config.target_modules}
        peft_model = get_peft_model(model, peft_config)
        if model_args.lora_init == "eva":
            device = next(peft_model.parameters()).device
            svd_dict = {f"{k}.lora_A.default.weight": v.to(device) for k, v in svd_dict.items()}
            if model_args.n_components_for_init:
                peft_sd = peft_model.base_model.model.state_dict()
                for k in list(svd_dict.keys()):
                    svd_dict[k] = torch.cat([
                        svd_dict[k][:model_args.n_components_for_init],
                        peft_sd[k][model_args.n_components_for_init:]
                    ], dim=0)
            peft_model.base_model.model.load_state_dict(svd_dict, strict=False)
            _ = [p.data.zero_() for n, p in peft_model.named_parameters() if "lora_B" in n]
        return peft_model
    return get_peft_model(model, peft_config)



def get_merged_model_and_tokenizer(cp_path, device="cpu", initial_adapter_weights_path=None):
    # load merged model from adapter checkpoint dir
    peft_config = PeftConfig.from_pretrained(cp_path)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model.to(device)
    init_lora_weights_orig = None
    if hasattr(peft_config, "init_lora_weights"):
        init_lora_weights_orig = peft_config.init_lora_weights
        peft_config.init_lora_weights = False
    model = PeftModel(model=model, peft_config=peft_config)
    is_pissa = str(init_lora_weights_orig).lower().startswith("pissa")
    is_olora = str(init_lora_weights_orig).lower() == "olora"
    if is_pissa or is_olora:
        if initial_adapter_weights_path is None:
            initial_adapter_weights_path = os.path.join(cp_path, "initial_adapter_model.safetensors")
        initial_adapter_sd = load_file(initial_adapter_weights_path, device=device) 
        for n in peft_config.target_modules:
            m = functools.reduce(getattr, n.split("."), model.base_model.model)
            initial_loraA_weight = initial_adapter_sd['base_model.model.' + n + '.lora_A.weight']
            initial_loraB_weight = initial_adapter_sd['base_model.model.' + n + '.lora_B.weight']
            new_weight = m.base_layer.weight.data - m.scaling["default"] * initial_loraB_weight @ initial_loraA_weight
            m.base_layer.weight.copy_(new_weight)
    adapter_sd = load_file(os.path.join(cp_path, "adapter_model.safetensors"))
    # make sure all possible keys can be mapped (e.g. dora magnitude vector does not end with .weight)
    key_map = {n.replace(".default", ""): n for n, m in model.named_parameters()}
    key_map.update({k.replace(".weight", ""): v for k, v in key_map.items()})
    adapter_sd = {key_map[k]: v for k, v in adapter_sd.items() if "base_layer" not in k}
    missing, unexpected = model.load_state_dict(adapter_sd, strict=False)
    model.merge_and_unload()
    model = model.base_model.model
    model.cpu()
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    return model, tokenizer


def match_module_name(module_name, name_to_match):
    return ".".join(module_name.split(".")[-name_to_match.count(".")-1:]) == name_to_match


def chunk_list(lst, n):
    """Yield n chunks from lst."""
    a, b = divmod(len(lst), n)
    chunk_lengths = [a + int(i<b) for i in range(n)]
    start = 0
    for l in chunk_lengths:
        yield lst[start:start+l]
        start += l


def find_equal_values(dictionary):
    value_dict = defaultdict(list)
    for k,v in dictionary.items():
        value_dict[v].append(k)
    return {k: v for k, v in value_dict.items() if len(v) > 1}


def get_hf_repo_ref(name):
    return name.split("/")[-1]


def get_svd_fileref(
    model_name,
    dataset_name,
    batch_size,
    seed,
    rank,
    rho,
    early_stop_sim_thresh,
    early_stop_redist_metric,
    scale_by_singular_values,
    whiten,
    use_label_mask,
    model_max_length
):
    model_ref = get_hf_repo_ref(model_name)
    data_ref = get_hf_repo_ref(dataset_name)
    fileref = f"{model_ref}_{data_ref}"
    if model_max_length:
        fileref += f"_len{model_max_length}"
    fileref += f"_r{rank}_rho{rho}_bs{batch_size}_seed{seed}_{early_stop_redist_metric}{early_stop_sim_thresh}"
    if scale_by_singular_values:
        fileref += "_scaled"
    if whiten:
        fileref += "_whitened"
    if use_label_mask:
        fileref += "_labelmask"
    return fileref


def get_wandb_run_name(
    model_name,
    dataset_name,
    adapter_type,
    lora_dim,
    lora_init,
    redistribute,
    svd_filepath,
    n_components_for_init
):
    model_ref = get_hf_repo_ref(model_name)
    data_ref = get_hf_repo_ref(dataset_name)
    run_name = f"{model_ref}_{data_ref}_{adapter_type}_r{lora_dim}_{lora_init}"
    if redistribute or lora_init == "eva":
        rho = re.search(r"rho\d+", svd_filepath).group(0)
        run_name += f"_{rho}"
        if "_scaled" in svd_filepath:
            run_name += "_scaled"
    if lora_init == "eva" and n_components_for_init:
            run_name += f"_n_init{n_components_for_init}"
    return run_name

"""
def get_valid_svd_filerefs(
    svd_path,
    model_name,
    dataset_name,
    lora_dim,
    rho,
    early_stop_p,
    early_stop_sim_thresh,
    scale_by_singular_values
):
    model_ref = get_hf_repo_ref(model_name)
    data_ref = get_hf_repo_ref(dataset_name)
    candidates = []
    for f in os.listdir(svd_path):
        if not "_svd.bin" in f:
            continue
        if not model_ref in f:
            continue
        if not data_ref in f:
            continue
        if not (m := re.search(r"r\d+", f)):
            continue
        if lora_dim * rho > int(m.group(0)[1:]):
            continue
        if not (p := re.search(r"\_p[0,1]\.\d+", f)):
            continue
        if float(p.group(0)[2:-1]) != early_stop_p:
            continue
        if not (s := re.search(r"\_sim[0,1]\.\d+\_", f)):
            continue
        if float(s.group(0)[4:-1]) != early_stop_sim_thresh:
            continue
        if scale_by_singular_values and not "_scaled" in f:
            continue
        candidates.append(f)
    return candidates
"""

def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        #pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

class NotEqual:
    def __eq__(self, other):
        return False
    

def cycle(iterable):
    while True:
        for x in iterable:
            yield x