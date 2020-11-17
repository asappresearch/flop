from typing import Dict, Any

import torch.nn as nn

try:
    import distiller
    from distiller.config import file_config, dict_config
except:
    print("distiller not installed.")


class NervanaPruner(object):
    def __init__(self, model: nn.Module, subpruners: Dict[str, Dict[str, Any]]):

        # Reorganize dictionary in Nervana format
        pruners = {}
        policies = []
        for name, kwargs in subpruners.items():

            # Split kwargs into pruner kwargs and policy kwargs
            pruner_kwargs = {}
            policy_kwargs = {"pruner": {"instance_name": name}}
            for key, value in kwargs.items():

                if key in {"starting_epoch", "ending_epoch", "epochs"}:
                    raise ValueError(
                        "Please provide arguments by step (e.g. `starting_step`, "
                        " `ending_step`, `steps`) instead of by epoch (e.g. "
                        "`starting_epoch`, `ending_epoch`, `epochs`)."
                    )

                # Search for policy kwargs
                if key == "starting_step":
                    policy_kwargs["starting_epoch"] = value
                elif key == "ending_step":
                    policy_kwargs["ending_epoch"] = value
                elif key == "steps":
                    policy_kwargs["steps"] = value
                elif key == "frequency":
                    policy_kwargs["frequency"] = value
                else:
                    pruner_kwargs[key] = value

            pruners[name] = pruner_kwargs
            policies.append(policy_kwargs)

        self.compression_scheduler = dict_config(
            model, None, {"pruners": pruners, "policies": policies}
        )

        # Verify that all weights marked for pruning exist in model
        model_param_names = set(n for n, _ in model.named_parameters())
        policies_by_step = self.compression_scheduler.policies.items()
        for step, policy_lst in policies_by_step:
            for policy in policy_lst:
                for name in policy.pruner.params_names:
                    if name not in model_param_names:
                        raise ValueError(
                            f"Weight `{name}` was marked for pruning at step {step}, but does not exist in model!"
                        )

    def begin_step(self, step: int):
        self.compression_scheduler.on_epoch_begin(step)

    def end_step(self, step: int):
        self.compression_scheduler.on_epoch_end(step)

    def begin_iter(self, step: int, n_iter: int, iter_per_step: int):
        self.compression_scheduler.on_minibatch_begin(
            step, minibatch_id=n_iter, minibatches_per_epoch=iter_per_step
        )

    def end_iter(self, step: int, n_iter: int, iter_per_step: int):
        self.compression_scheduler.on_minibatch_end(
            step, minibatch_id=n_iter, minibatches_per_epoch=iter_per_step
        )

    def get_step_logs(self):
        model = self.compression_scheduler.model
        t, total = distiller.weights_sparsity_tbl_summary(
            model, return_total_sparsity=True
        )
        return {"sparsity": total / 100}
