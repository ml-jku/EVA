import time
import torch
from tqdm import tqdm
from functools import reduce
from collections import Counter
from torch_incremental_pca import IncrementalPCA

from src.utils import match_module_name, find_equal_values, cycle

from typing import Union


class SVDHook:
    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor]
    ):
        self.name = name
        self.n_components = n_components
        self.sim_thresh = sim_thresh

        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            assert check1 and check2, "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"

        self.svd = IncrementalPCA(n_components=n_components, copy=True, lowrank=True)

        self.indices = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()

        try:
            states = input.detach()
        except AttributeError:
            states = input[0].detach()
        states = states[self.indices[:, 0], self.indices[:, 1], :]

        if states.size(0) < self.n_components:
            return

        self.svd.partial_fit(states.to(torch.float32))

        if previous_components is not None:
            components = self.svd.components_
            if len(components.shape) == 1:
                components = components.reshape(1, -1)
                previous_components = previous_components.reshape(1, -1)
            # consider as converged if enough components have converged via cossim
            sim = torch.nn.functional.cosine_similarity(components, previous_components)
            self.converged = (sim >= self.sim_thresh)


class HashHook:

    def __init__(self, name: str):
        self.name = name
        self.hashed_inputs = []

    @staticmethod
    def hash_fn(tensor):
        return hash(tuple(tensor.view(-1).tolist()))

    def __call__(self, model, input, output):
        try:
            x = input.detach().cpu()
        except AttributeError:
            x = input[0].detach().cpu()
        self.hashed_inputs.append(self.hash_fn(x))


@torch.no_grad()
def compute_svd(
    model,
    data_loader,
    rank,
    rho=2,
    early_stop_sim_thresh=0.99,
    early_stop_redist_metric="ratio",
    scale_by_singular_values=False,
    whiten=False,
    target_modules=None,
    ignore_modules=None,
    use_label_mask=True,
    min_batches=1,
    log_convergence_stats=False
):

    def _get_metric(svd, metric):
        if metric == "raw":
            return svd.explained_variance_
        elif metric == "ratio":
            return svd.explained_variance_ratio_
        elif metric == "sum":
            return svd.explained_variance_ / svd.explained_variance_.sum()
        elif metric == "max":
            return svd.explained_variance_ / svd.explained_variance_.max()

        else:
            raise ValueError(f"Invalid metric: {metric}")
        
    def _get_rank_distribution(hooks, hook_layer_map, equal_inputs_map, metric, rank_budget, max_components):
        exp_vars = {k: _get_metric(h.svd, metric)[:max_components] for k, h in hooks.items()}
        keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in exp_vars[name]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[:rank_budget]])
        counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()} # add layers with 0 rank
        for k, k_hook in equal_inputs_map.items():
            # ensure hook layers have the highest rank if they are equal to another layer
            rank, rank_hook = counts[k], counts[k_hook]
            if rank_hook >= rank:
                continue
            counts[k_hook], counts[k] = rank, rank_hook
        return counts

    assert rho >= 1, "early_stop_rho must be >= 1"
    max_components = round(rank * rho)
    device = next(model.parameters()).device
    training = model.training
    model.eval()

    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if target_modules:
                check = [match_module_name(name, t) for t in target_modules]
                if not any(check):
                    continue
            if ignore_modules:
                check = [match_module_name(name, i) for i in ignore_modules]
                if any(check):
                    continue
            hook = HashHook(name)
            module.register_forward_hook(hook)
            hooks[name] = hook
    rank_budget = len(hooks) * rank

    # forward for one batch to check which layer inputs are equal to avoid unneeded svd calculations
    inputs = {k: v.to(device) for k, v in next(iter(data_loader)).items() if k != "labels"}
    model(**inputs)
    hash_dict = {k: h.hashed_inputs[0] for k, h in hooks.items()}
    equal_inputs_map = {vv: v[0] for v in find_equal_values(hash_dict).values() for vv in v[1:]}
    hooks = {k: SVDHook(k, max_components, early_stop_sim_thresh) for k in hooks.keys() if k not in equal_inputs_map}
    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}
    for name in layer_hook_map.keys():
        module = reduce(getattr, name.split("."), model)
        module._forward_hooks.clear()

    has_converged_stats = None
    if log_convergence_stats:
        has_converged_stats = [{
            "rank": rank,
            "rho": rho,
            "early_stop_sim_thresh": early_stop_sim_thresh,
            "early_stop_redist_metric": early_stop_redist_metric,
            "scale_by_singular_values": scale_by_singular_values,
            "whiten": whiten,
            "target_modules": target_modules,
            "ignore_modules": ignore_modules,
            "equal_inputs_map": equal_inputs_map
        }]
    
    # start svd calculation
    pbar = tqdm(enumerate(iter(cycle(data_loader))), position=0, leave=False)
    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = {k: max_components for k in layer_hook_map.keys()}
    for i, inputs in pbar:

        t0 = time.perf_counter()

        mask = inputs["attention_mask"]
        if use_label_mask:
            mask = torch.logical_and(mask.bool(), inputs["labels"] != -100)
        indices = torch.nonzero(mask)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}

        for name, hook in hooks.items():
            module = reduce(getattr, name.split("."), model)
            module._forward_hooks.clear()
            # check if all components that are needed for the rank distribution have converged
            if torch.all(hook.converged[:rank_dist[name]]):
                convergence_dict[name] = True
                continue
            convergence_dict[name] = False
            hook.indices = indices
            module.register_forward_hook(hook)

        if all(convergence_dict.values()) and i > min_batches:
            print("exiting - all svd components have converged.")
            break

        model(**inputs)

        # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of components
        if not all([hasattr(h.svd, "components_") for h in hooks.values()]):
            continue

        rank_dist = _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, early_stop_redist_metric, rank_budget, max_components)

        step_time = time.perf_counter() - t0

        layer_converged = list(convergence_dict.values()) + [convergence_dict[v] for v in equal_inputs_map.values()]
        pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

        if log_convergence_stats:
            stats = {k: hook.converged.tolist() for k, hook in hooks.items()}
            has_converged_stats.append((stats, step_time))

    svd_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        hook = hooks[layer_hook_map[name]]
        assert torch.all(hook.converged[:rank]) # this should never happen because we check for convergence
        u = hook.svd.components_[:rank]
        if whiten:
            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
        elif scale_by_singular_values:
            s = hook.svd.singular_values_[:rank]
            s /= s.max()
            u *= s.reshape(-1, 1)
        svd_dict[name] = u

    # objects are torch tensors on the model device
    svd_dict = {k: v.cpu() for k, v in svd_dict.items()}

    # restore model state
    model.train(training)

    return svd_dict, has_converged_stats