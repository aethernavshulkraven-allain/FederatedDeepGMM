import copy
import json
import logging
import math
import csv
import os
import time
import numpy 
import torch
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from .multiprocess_client import (
    MultiprocessClientExecutor,
    SingleGPUMultiprocessClientExecutor,
    _to_cpu,
    _to_device,
)
from .single_gpu_client import SingleGPUClientExecutor

# from fedml.core.dp.mechanisms import 
# from opacus.layers import 
from model_selection.simple_model_eval import GradientDecentSimpleModelEval
from model_selection_class import FHistoryModelSelectionV3
from game_objectives.simple_moment_objective import (
    OptimalMomentObjective,
    PaperAlignedMomentObjective,
)
from optimizers.oadam import OAdam
from optimizers.Customsgd import CustomSGD
from optimizers.ogda import OGDA
from optimizers.optimizer_factory import OptimizerFactory
# from optimizers.optimizer_factory import DPOAdam
from torch.optim import Adam,sgd
from model_selection.simple_model_eval import SGDSimpleModelEval
from model_selection.learning_eval_nostop import \
    FHistoryLearningEvalSGDNoStop
from game_objectives.approximate_psi_objective import approx_psi_eval
# from fedgmm.sp_decentralized_mnist_lr_example.plotting import PlotElement
from plotting import PlotElement
import matplotlib.pyplot as plt
from experiment_utils import (
    AGGREGATION_WEIGHTING_CHOICES,
    OBJECTIVE_MODE_CHOICES,
    PAPER_ALIGNED_LAMBDA,
    append_aggregation_weights_by_round,
    append_mse_by_round,
    append_test_mse_by_round,
    check_eicu_aggregation_weighting,
    check_eicu_objective_mode,
    client_indices_for_round,
    compute_client_weights,
    config_checksum,
    default_critic_multiplier,
    env_truthy,
    ensure_run_dirs,
    equal_client_moment_violation,
    equal_client_structural_mse,
    get_effective_config,
    moment_violation,
    metric_values_are_finite,
    prepare_run_dir,
    predictions_for_split,
    run_dir_from_config,
    RuntimeProfiler,
    save_predictions_npz,
    state_is_finite,
    structural_mse,
    structural_mse_from_predictions,
    now_seconds,
    weighted_average_state_dicts,
    write_effective_config,
    write_aggregation_weights_by_round,
    write_batching_summary,
    write_metrics,
    write_mse_by_round,
    write_per_client_metrics,
    write_test_mse_by_round,
)


def prepare_curve_data(dataset, predictions, targets):
    """Return scalar coordinates and aligned values for curve output.

    Image-treatment scenarios keep their scalar causal coordinate in ``w``.
    Sorting the image tensor itself creates one index per pixel and can expand
    the result arrays by several orders of magnitude.
    """
    x = dataset.x.detach().cpu().numpy()
    coordinate = dataset.w.detach().cpu().numpy() if x.ndim > 2 else x
    coordinate = numpy.asarray(coordinate).reshape(coordinate.shape[0], -1)
    if coordinate.shape[1] != 1:
        raise ValueError(
            "Curve plotting requires one scalar coordinate per observation; "
            f"received shape {coordinate.shape}."
        )

    indices = numpy.argsort(coordinate[:, 0])
    return coordinate[indices], predictions[indices], targets[indices]


def log_results_to_csv(file_path, round_number, mse):
       if os.path.dirname(file_path):
           os.makedirs(os.path.dirname(file_path), exist_ok=True)
       file_exists = os.path.isfile(file_path)
       with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Round', 'MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Round': round_number, 'MSE': mse}) 
class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        # Defensive initialization of research-specific arguments
        research_args = {
            'video_plotter': False,
            'print_freq': 1,
            'eval_freq': 1,
            'burn_in': 0,
            'max_no_progress': 10000,
            'verbose': True,
            'print_freq_mul': 1,
            'server_learning_rate': 1.5,
            'eg_predictor_server_lr': None,
            'eg_corrector_server_lr': None,
            'zo_mu': 1e-3,
            'zo_num_directions': 1,
            'enable_multiprocessing': False,
            'multiprocessing_num_workers': 0,
            'multiprocessing_gpu_ids': None,
            'client_execution_mode': None,
            'single_gpu_num_slots': 0,
            'single_gpu_id': None,
            'multiprocessingsinglegpu_num_workers': 2,
            'multiprocessingsinglegpu_gpu_id': None,
        }
        for arg_name, default_value in research_args.items():
            if not hasattr(self.args, arg_name):
                setattr(self.args, arg_name, default_value)
        if not hasattr(self.args, "critic_multiplier"):
            setattr(self.args, "critic_multiplier", default_critic_multiplier(args.dataset))
        if not hasattr(self.args, "server_learning_rate"):
            setattr(self.args, "server_learning_rate", 1.5)
        if not hasattr(self.args, "objective_lambda_1"):
            setattr(self.args, "objective_lambda_1", 0.1)
        if not hasattr(self.args, "aggregation_weighting"):
            setattr(self.args, "aggregation_weighting", "sample_size")
        if not hasattr(self.args, "objective_mode"):
            setattr(self.args, "objective_mode", "legacy")
        if not hasattr(self.args, "campaign_role"):
            setattr(self.args, "campaign_role", "")
        if not hasattr(self.args, "output_dir"):
            setattr(self.args, "output_dir", "results")
        if not hasattr(self.args, "run_id"):
            setattr(self.args, "run_id", "0")
        if not hasattr(self.args, "random_seed"):
            setattr(self.args, "random_seed", 0)
        if not hasattr(self.args, "gradient_clip_norm"):
            setattr(self.args, "gradient_clip_norm", 1.0)
        if not hasattr(self.args, "simple_model_selection_epochs"):
            setattr(self.args, "simple_model_selection_epochs", 100)
        if not hasattr(self.args, "f_history_model_selection_epochs"):
            setattr(self.args, "f_history_model_selection_epochs", 60)
        if not hasattr(self.args, "model_selection_batch_size"):
            setattr(self.args, "model_selection_batch_size", 200)
        if not hasattr(self.args, "model_selection_max_samples"):
            setattr(self.args, "model_selection_max_samples", 0)
        if not hasattr(self.args, "skip_model_selection"):
            setattr(self.args, "skip_model_selection", False)
        if not hasattr(self.args, "skip_gmm_eval"):
            setattr(self.args, "skip_gmm_eval", bool(getattr(self.args, "skip_model_selection", False)))
        if not hasattr(self.args, "gmm_eval_proxy"):
            setattr(self.args, "gmm_eval_proxy", "negative_val_mse" if bool(getattr(self.args, "skip_gmm_eval", False)) else "approx_psi")
        if bool(getattr(self.args, "skip_model_selection", False)):
            setattr(self.args, "skip_gmm_eval", True)
            setattr(self.args, "gmm_eval_proxy", "negative_val_mse")
        if not hasattr(self.args, "auxiliary_regression"):
            setattr(self.args, "auxiliary_regression", True)
        if not hasattr(self.args, "auxiliary_regression_epochs"):
            setattr(self.args, "auxiliary_regression_epochs", int(getattr(self.args, "epochs", 0)))
        if not hasattr(self.args, "auxiliary_regression_state_device"):
            setattr(self.args, "auxiliary_regression_state_device", "device")
        if not hasattr(self.args, "append_round_csv"):
            setattr(self.args, "append_round_csv", True)
        if not hasattr(self.args, "periodic_checkpoint_interval"):
            setattr(self.args, "periodic_checkpoint_interval", 200)
        if not hasattr(self.args, "enable_legacy_outputs"):
            setattr(self.args, "enable_legacy_outputs", True)
        if not hasattr(self.args, "overwrite"):
            setattr(self.args, "overwrite", False)
        if not hasattr(self.args, "require_multibatch_stochastic"):
            setattr(self.args, "require_multibatch_stochastic", False)
        if not hasattr(self.args, "log_test_mse_by_round"):
            setattr(self.args, "log_test_mse_by_round", False)
        if not hasattr(self.args, "test_mse_used_for_selection"):
            setattr(self.args, "test_mse_used_for_selection", False)
        if not hasattr(self.args, "selection_metric_source"):
            setattr(self.args, "selection_metric_source", "validation")
        if bool(getattr(self.args, "test_mse_used_for_selection", False)):
            raise ValueError("test_mse_used_for_selection must remain false")
        if str(getattr(self.args, "selection_metric_source", "validation")).lower() != "validation":
            raise ValueError("selection_metric_source must be validation")
        _aggregation_weighting = str(getattr(self.args, "aggregation_weighting", "sample_size"))
        if _aggregation_weighting not in AGGREGATION_WEIGHTING_CHOICES:
            raise ValueError(
                f"aggregation_weighting must be one of {AGGREGATION_WEIGHTING_CHOICES}, "
                f"got {_aggregation_weighting!r}"
            )
        _objective_mode = str(getattr(self.args, "objective_mode", "legacy"))
        if _objective_mode not in OBJECTIVE_MODE_CHOICES:
            raise ValueError(
                f"objective_mode must be one of {OBJECTIVE_MODE_CHOICES}, "
                f"got {_objective_mode!r}"
            )
        _campaign_role = str(getattr(self.args, "campaign_role", ""))
        check_eicu_aggregation_weighting(
            getattr(self.args, "dataset", ""), _aggregation_weighting, _campaign_role
        )
        check_eicu_objective_mode(getattr(self.args, "dataset", ""), _objective_mode)
        if bool(getattr(self.args, "require_multibatch_stochastic", False)) and int(getattr(self.args, "batch_size", 0)) <= 0:
            raise ValueError("require_multibatch_stochastic requires a positive batch_size")
        self.effective_config = get_effective_config(self.args)
        self.run_dir = run_dir_from_config(self.effective_config)
        self.profiler = getattr(self.args, "_fedgmm_runtime_profiler", None)
        if self.profiler is None:
            self.profiler = RuntimeProfiler.from_config(self.effective_config)
            setattr(self.args, "_fedgmm_runtime_profiler", self.profiler)
            self.profiler.start_telemetry()
        self.profiler.configure(self.effective_config, self.run_dir)
        self.profiler.record_environment(self.effective_config, self.run_dir, self.device)
        self.profile_skip_aux_reg = (
            bool(getattr(self.profiler, "enabled", False))
            and env_truthy("FEDGMM_PROFILE_SKIP_AUX_REG", default=False)
        )
        self.auxiliary_regression_enabled = bool(self.effective_config.get("auxiliary_regression", True))
        self.skip_auxiliary_regression = self.profile_skip_aux_reg or not self.auxiliary_regression_enabled
        self.append_round_csv = bool(self.effective_config.get("append_round_csv", True))
        self.periodic_checkpoint_interval = int(self.effective_config.get("periodic_checkpoint_interval", 200))
        self.skip_model_selection = bool(self.effective_config.get("skip_model_selection", False))
        self.skip_gmm_eval = bool(self.effective_config.get("skip_gmm_eval", self.skip_model_selection))
        self.gmm_eval_proxy = str(self.effective_config.get("gmm_eval_proxy", "negative_val_mse" if self.skip_gmm_eval else "approx_psi"))
        if self.skip_auxiliary_regression:
            self.profiler.record_once(
                "skip_auxiliary_regression",
                "skip_auxiliary_regression",
                detail=(
                    "Auxiliary regression train/aggregate/set disabled; "
                    f"profile_env={self.profile_skip_aux_reg}; config_enabled={self.auxiliary_regression_enabled}"
                ),
            )
        if self.skip_model_selection:
            self.profiler.record_once(
                "skip_model_selection",
                "skip_model_selection",
                detail="Fast final mode: model-selection critic history is bypassed and gmm_eval uses proxy diagnostics.",
            )
        with self.profiler.span("prepare_run_dir"):
            prepare_run_dir(self.run_dir, overwrite=bool(getattr(self.args, "overwrite", False)))
            ensure_run_dirs(self.run_dir)
            write_effective_config(self.run_dir, self.effective_config)
        self.server_learning_rate = float(self.effective_config["server_learning_rate"])
        self.critic_multiplier = float(self.effective_config["critic_multiplier"])
        self.objective_lambda_1 = float(self.effective_config["objective_lambda_1"])
        self.aggregation_weighting = str(self.effective_config["aggregation_weighting"])
        self.objective_mode = str(self.effective_config["objective_mode"])
        self.mse_rows = []
        self.test_mse_rows = []
        self.aggregation_weight_rows = []
        self.log_test_mse_by_round = bool(self.effective_config.get("log_test_mse_by_round", False))
        self.best_validation_mse = float("inf")
        self.best_validation_round = None
        self.best_g_state = None
        self.best_f_state = None
        self.best_reg_state = None
        # Second, independent checkpoint: lowest held-out moment violation.
        # This is the g0-free selection criterion (works even without ground
        # truth, unlike best_validation_mse above), tracked separately so both
        # can be compared -- the two need not select the same round.
        self.best_moment_violation = float("inf")
        self.best_moment_violation_round = None
        self.best_moment_state = None
        self.final_g_state = None
        self.final_f_state = None
        self.final_reg_state = None
        self.best_state = None
        self.final_state = None
        self.delta_g_prev = None
        self.delta_f_prev = None
        self.diverged = False
        self.nonfinite_first_round = None
        self.batching_summary_rows = []
        self.start_time = now_seconds()
        [
        train_data_num,
        test_data_num,
        val_data_num,
        train_data_global,
        test_data_global,
        val_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        class_num, 
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = val_data_global
        for split_name, split in [("train", self.train_global), ("val", self.val_global), ("test", self.test_global)]:
            if not hasattr(split, "g") or split.g is None:
                raise ValueError(f"{split_name} split does not expose ground-truth structural g")
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.val_data_num_in_total = val_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict


        # The FedAvg parallel-execution path (multi/single-GPU client
        # executors) only carries g and f between server and clients, so it
        # can't service the auxiliary regression model. Only drop the legacy
        # direct-regression model when that diagnostic path is actually off;
        # eICU/Study A runs set auxiliary_regression=true and need model[2].
        if not bool(self.args.auxiliary_regression):
            model = model[:2]
        logging.info("model = {}".format(model))
        self.model = model
        self.device = device

        # Move models to device
        with self.profiler.span("model_to_device"):
            if isinstance(self.model, list):
                for model_list in self.model:
                    if isinstance(model_list, list):
                        for m in model_list:
                            if isinstance(m, torch.nn.Module):
                                m.to(self.device).double()
                    elif isinstance(model_list, torch.nn.Module):
                        model_list.to(self.device).double()
            elif isinstance(self.model, torch.nn.Module):
                self.model.to(self.device).double()

        # FedML's resolved device is authoritative. Keep every global
        # split on the same device as the models before model selection.
        with self.profiler.span("global_tensor_to_device"):
            for split in (self.train_global, self.val_global, self.test_global):
                for tensor_name in ("x", "z", "y", "g", "w"):
                    tensor = getattr(split, tensor_name, None)
                    if torch.is_tensor(tensor):
                        setattr(
                            split,
                            tensor_name,
                            tensor.to(self.device),
                        )
        with self.profiler.span("eval_target_cache"):
            self._train_y_cpu = self.train_global.y.detach().cpu()
            self._train_g_cpu = self.train_global.g.detach().cpu()
            self._val_y_cpu = self.val_global.y.detach().cpu()
            self._val_g_cpu = self.val_global.g.detach().cpu()
            self._test_g_cpu = self.test_global.g.detach().cpu()
            self._test_y_cpu = self.test_global.y.detach().cpu()

        # g_learning_rates = [0.010, 0.050, 0.020]
        ##g_learning_rates =[0.01, 0.001,0.0001,0.0005]
        # g_learning_rates = [0.0005]
        # g_learning_rates=[0.1,0.2,0.5]
        # g_learning_rates =[0.00010, 0.000050, 0.000020]
        # Use the learning rate from the YAML configuration
        g_learning_rates = [self.args.learning_rate]
        if self.objective_mode == "paper_aligned":
            # lambda is fixed at 1/4 for this mode -- not objective_lambda_1,
            # which only tunes the legacy objective. See PaperAlignedMomentObjective.
            game_objectives = [
                PaperAlignedMomentObjective(lambda_1=PAPER_ALIGNED_LAMBDA),
            ]
        else:
            game_objectives = [
                OptimalMomentObjective(lambda_1=self.objective_lambda_1),
            ]
        learning_setups = []
        critic_multiplier = self.critic_multiplier
       
        for g_lr in g_learning_rates:
            for game_objective in game_objectives:
                # learning_setup = {
                #     "g_optimizer_factory": OptimizerFactory(
                #         CustomSGD, lr=float(g_lr), betas=(0.5,0.9)),
                #     "f_optimizer_factory": OptimizerFactory(
                #         CustomSGD, lr=1000*float(g_lr),betas =(0.5,0.9)),
                #     "game_objective": game_objective
                # }
                # learning_setups.append(learning_setup)
                if args.client_optimizer == "ogda":
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            OGDA, lr=float(g_lr)),
                        "f_optimizer_factory": OptimizerFactory(
                            OGDA, lr=critic_multiplier*float(g_lr)),
                        "game_objective": game_objective
                    }
                else:
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            CustomSGD, lr=float(g_lr), momentum=0.0),
                        "f_optimizer_factory": OptimizerFactory(
                            CustomSGD, lr=critic_multiplier*float(g_lr), momentum=0.0),
                        "game_objective": game_objective
                    }
                learning_setups.append(learning_setup)
        
#              learning_setup = {
#                       "g_optimizer_factory": OptimizerFactory(
#                        CustomSGD, lr=float(g_lr), momentum=0.9),  # Using SGD with momentum
#                       "f_optimizer_factory": OptimizerFactory(
#                        CustomSGD, lr=5.0*float(g_lr), momentum=0.9),  # Note the increased learning rate for f_optimizer
#                       "game_objective": game_objective
# }
        if args.client_optimizer == "ogda":
            default_g_opt_factory = OptimizerFactory(OGDA, lr=0.01)
            # default_f_opt_factory = OptimizerFactory(OGDA, lr=0.01)
            default_f_opt_factory = OptimizerFactory(OGDA, lr=critic_multiplier*args.learning_rate)
        else:
            default_g_opt_factory = OptimizerFactory(CustomSGD, lr=0.01, momentum=0.0)
            # default_f_opt_factory = OptimizerFactory(CustomSGD, lr=0.01, momentum=0.0)
            default_f_opt_factory = OptimizerFactory(CustomSGD, lr=critic_multiplier*args.learning_rate, momentum=0.0)
            
        # default_g_opt_factory = OptimizerFactory(
        #     sgd, lr=0.0001, betas=(0.5, 0.9))
        # default_f_opt_factory = OptimizerFactory(
        #     sgd, lr=0.001, betas=(0.5, 0.9))
        # default_g_opt_factory = OptimizerFactory(
        #     CustomSGD, lr=0.01, momentum=0.9)
        # default_f_opt_factory = OptimizerFactory(
        #     CustomSGD, lr=0.01, momentum=0.9)
        # g_simple_model_eval = GradientDecentSimpleModelEval(
        #     max_num_iter=100, max_no_progress=10, eval_freq=1)      
        g_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=int(self.args.simple_model_selection_epochs),
            max_no_progress=10,
            batch_size=int(self.args.model_selection_batch_size),
            eval_freq=1)
        f_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=int(self.args.simple_model_selection_epochs),
            max_no_progress=10,
            batch_size=int(self.args.model_selection_batch_size),
            eval_freq=1)
        learning_eval = FHistoryLearningEvalSGDNoStop(
            num_epochs=int(self.args.f_history_model_selection_epochs),
            eval_freq=1,
            batch_size=int(self.args.model_selection_batch_size))
        psi_eval_burn_in = min(
            30,
            max(0, int(self.args.f_history_model_selection_epochs) - 1),
        )
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=model[0],
            f_model_list=model[1],
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_max_no_progress=10, psi_eval_burn_in=psi_eval_burn_in,
            failure_context={
                "dataset": self.args.dataset,
                "random_seed": self.args.random_seed,
                "simple_model_selection_epochs": self.args.simple_model_selection_epochs,
                "f_history_model_selection_epochs": self.args.f_history_model_selection_epochs,
                "model_selection_batch_size": self.args.model_selection_batch_size,
                "learning_rate": self.args.learning_rate,
                "critic_multiplier": self.critic_multiplier,
                "f_learning_rate": self.critic_multiplier * float(self.args.learning_rate),
                "client_optimizer": self.args.client_optimizer,
            },
        )
        self.default_g_opt_factory = default_g_opt_factory
        # g_simple_model_eval = SGDSimpleModelEval()
        # f_simple_model_eval = SGDSimpleModelEval()
        # learning_eval = FHistoryLearningEvalSGDNoStop(num_epochs=args.epochs_model_selection, eval_freq=args.eval_freq, print_freq=args.print_freq, batch_size=args.batch_size)
        # self.model_selection = FHistoryModelSelectionV3(
        #     g_model_list=model[0],
        #     f_model_list=model[1],
        #     learning_args_list=learning_setups,
        #     default_g_optimizer_factory=default_g_opt_factory,
        #     default_f_optimizer_factory=default_f_opt_factory,
        #     g_simple_model_eval=g_simple_model_eval,
        #     f_simple_model_eval=f_simple_model_eval,
        #     learning_eval=learning_eval,
        #     psi_eval_max_no_progress=self.args.psi_eval_max_no_progress, psi_eval_burn_in=self.args.psi_eval_burn_in)
        # model_linear_sgd_fedavg = torch.load('/home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/model_linear_fedsgd')
        # fedavg_sgd = model_linear_sgd_fedavg(self.test_global.x)
        # sgd_plain = model_linear_sgd_plain(self.test_global.x)
        # mse = float(((fedavg_sgd - self.test_global.g) ** 2).mean())
        model_selection_max_samples = int(self.args.model_selection_max_samples)
        if model_selection_max_samples > 0:
            train_limit = min(model_selection_max_samples, int(train_data_global.y.shape[0]))
            dev_limit = min(model_selection_max_samples, int(val_data_global.y.shape[0]))
            # A capped model-selection critic history has ``dev_limit`` rows.
            # Keep the smoke-only validation view aligned with that history so
            # downstream approximate-psi evaluation compares like-sized data.
            for tensor_name in ("x", "z", "y", "g", "w"):
                tensor = getattr(val_data_global, tensor_name, None)
                if torch.is_tensor(tensor):
                    setattr(val_data_global, tensor_name, tensor[:dev_limit])
            self.val_global = val_data_global
            self.val_data_num_in_total = dev_limit
            x_train_selection = train_data_global.x[:train_limit]
            z_train_selection = train_data_global.z[:train_limit]
            y_train_selection = train_data_global.y[:train_limit]
            x_dev_selection = val_data_global.x
            z_dev_selection = val_data_global.z
            y_dev_selection = val_data_global.y
        else:
            x_train_selection = train_data_global.x
            z_train_selection = train_data_global.z
            y_train_selection = train_data_global.y
            x_dev_selection = val_data_global.x
            z_dev_selection = val_data_global.z
            y_dev_selection = val_data_global.y
        if self.skip_model_selection:
            if len(model[0]) != 1 or len(model[1]) != 1 or len(learning_setups) != 1:
                raise ValueError(
                    "skip_model_selection is only valid when the generated config has exactly "
                    "one g model, one f model, and one learning setup"
                )
            with self.profiler.span("model_selection_fast_init"):
                g_global = model[0][0]
                f_global = model[1][0]
                g_global.initialize()
                f_global.initialize()
                learning_args = learning_setups[0]
                dev_f_collection = []
                e_dev_tilde = torch.zeros_like(y_dev_selection.detach().reshape(-1))
            self.skip_gmm_eval = True
            self.gmm_eval_proxy = "negative_val_mse"
        else:
            with self.profiler.span("model_selection"):
                g_global, f_global, learning_args, dev_f_collection, e_dev_tilde = \
                    self.model_selection.do_model_selection(
                        x_train=x_train_selection, z_train=z_train_selection, y_train=y_train_selection,
                        x_dev=x_dev_selection, z_dev=z_dev_selection, y_dev=y_dev_selection, verbose=True)
        
        self.eval_history = []
        self.g_state_history = []
        self.epsilon_dev_history = []
        self.epsilon_train_history = []

        self.g_of_x_train_list = []
        self.g_of_x_dev_list = []

        self.mse_list = []
        self.eval_list = []
        self.dev_f_collection = dev_f_collection
        self.e_dev_tilde = e_dev_tilde
        
        with self.profiler.span("create_model_trainer"):
            model_trainer_inputs = [g_global, f_global]
            if not self.skip_auxiliary_regression:
                model_trainer_inputs.append(model[2][0])
            self.model_trainer = create_model_trainer(model_trainer_inputs, learning_args, args)

        self.client_executor = self._create_client_executor()
        if self.client_executor is not None and not self.skip_auxiliary_regression:
            # The parallel client executors only ship g/f between server and
            # workers (see _run_primary_client_updates below), so they can't
            # service the auxiliary regression model.
            raise ValueError(
                "auxiliary_regression is not supported together with a "
                "parallel client_execution_mode; set auxiliary_regression: "
                "false or use client_execution_mode: sp"
            )
        if self.client_executor is None:
            with self.profiler.span("setup_clients"):
                self._setup_clients(
                    train_data_local_num_dict, train_data_local_dict,
                    test_data_local_dict, self.model_trainer,
                )

    def _create_client_executor(self):
        execution_mode = self.args.client_execution_mode
        if execution_mode is None or not str(execution_mode).strip():
            execution_mode = (
                "multi_gpu_processes"
                if self.args.enable_multiprocessing else "sp"
            )
        execution_mode = str(execution_mode).strip().lower()
        valid_modes = {
            "sp",
            "multi_gpu_processes",
            "single_gpu_streams",
            "multiprocessingsinglegpu",
        }
        if execution_mode not in valid_modes:
            raise ValueError(
                f"Unknown client_execution_mode {execution_mode!r}; "
                f"expected one of {sorted(valid_modes)}"
            )
        self.client_execution_mode = execution_mode

        if execution_mode == "sp":
            return None
        if not torch.cuda.is_available():
            logging.warning(
                "%s requested without CUDA; using SP execution",
                execution_mode,
            )
            self.client_execution_mode = "sp"
            return None

        if execution_mode == "multiprocessingsinglegpu":
            gpu_id = self.args.multiprocessingsinglegpu_gpu_id
            if gpu_id is None:
                gpu_id = self.device.index
                if gpu_id is None:
                    gpu_id = torch.cuda.current_device()
            gpu_id = int(gpu_id)
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid multiprocessingsinglegpu_gpu_id {gpu_id}; "
                    f"PyTorch sees {torch.cuda.device_count()} GPUs"
                )
            coordinator_id = self.device.index
            if coordinator_id is None:
                coordinator_id = torch.cuda.current_device()
            if gpu_id != coordinator_id:
                raise ValueError(
                    "multiprocessingsinglegpu_gpu_id must match "
                    "device_args.gpu_id so the entire run stays on one GPU"
                )
            worker_count = min(
                int(self.args.multiprocessingsinglegpu_num_workers),
                int(self.args.client_num_per_round),
            )
            if worker_count < 2:
                logging.warning(
                    "Single-GPU multiprocessing requires at least two "
                    "workers; using SP execution"
                )
                self.client_execution_mode = "sp"
                return None
            return SingleGPUMultiprocessClientExecutor(
                self.model_trainer,
                self.args,
                gpu_id,
                worker_count,
            )

        if execution_mode == "single_gpu_streams":
            gpu_id = self.args.single_gpu_id
            if gpu_id is None:
                gpu_id = self.device.index
                if gpu_id is None:
                    gpu_id = torch.cuda.current_device()
            gpu_id = int(gpu_id)
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid single_gpu_id {gpu_id}; PyTorch sees "
                    f"{torch.cuda.device_count()} GPUs"
                )
            coordinator_id = self.device.index
            if coordinator_id is None:
                coordinator_id = torch.cuda.current_device()
            if gpu_id != coordinator_id:
                raise ValueError(
                    "single_gpu_id must match device_args.gpu_id so the "
                    "entire run stays on one GPU"
                )
            slot_count = min(
                int(self.args.single_gpu_num_slots),
                int(self.args.client_num_per_round),
            )
            if slot_count < 2:
                logging.warning(
                    "Single-GPU concurrency requires at least two slots; "
                    "using SP execution"
                )
                self.client_execution_mode = "sp"
                return None
            return SingleGPUClientExecutor(
                self.model_trainer,
                self.args,
                torch.device("cuda", gpu_id),
                slot_count,
            )

        gpu_ids = self.args.multiprocessing_gpu_ids
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, str):
            gpu_ids = [
                int(value.strip())
                for value in gpu_ids.split(",") if value.strip()
            ]
        else:
            gpu_ids = [int(value) for value in gpu_ids]

        invalid_ids = [
            value for value in gpu_ids
            if value < 0 or value >= torch.cuda.device_count()
        ]
        if invalid_ids:
            raise ValueError(
                f"Invalid multiprocessing_gpu_ids {invalid_ids}; "
                f"PyTorch sees {torch.cuda.device_count()} GPUs"
            )
        worker_limit = int(self.args.multiprocessing_num_workers)
        if worker_limit > 0:
            gpu_ids = gpu_ids[:worker_limit]
        gpu_ids = gpu_ids[:self.args.client_num_per_round]
        if len(gpu_ids) < 2:
            logging.warning(
                "Multiprocessing requires at least two GPU workers; using SP execution"
            )
            self.client_execution_mode = "sp"
            return None
        return MultiprocessClientExecutor(self.model_trainer, self.args, gpu_ids)

    @staticmethod
    def _materialize_client_data(data_loader):
        return [_to_cpu(batch) for batch in data_loader]

    def _run_primary_client_updates(self, client_indexes, g_global, f_global):
        phase_start = time.perf_counter()
        if self.client_executor is None:
            w_locals = []
            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )
                weights = client.train(copy.deepcopy(g_global), copy.deepcopy(f_global))
                sample_number = client.get_sample_number()
                w_locals.append((sample_number, copy.deepcopy(weights)))
            logging.info(
                "Client primary phase mode=sp clients=%d elapsed_seconds=%.6f",
                len(client_indexes), time.perf_counter() - phase_start,
            )
            return w_locals

        tasks = []
        cpu_g_global = _to_cpu(g_global)
        cpu_f_global = _to_cpu(f_global)
        for client_idx in client_indexes:
            tasks.append(
                {
                    "phase": "primary",
                    "client_idx": int(client_idx),
                    "train_data": self._materialize_client_data(
                        self.train_data_local_dict[client_idx]
                    ),
                    "sample_number": self.train_data_local_num_dict[client_idx],
                    "g_global": cpu_g_global,
                    "f_global": cpu_f_global,
                }
            )
        results = self.client_executor.run(tasks)
        w_locals = []
        for client_idx, result in zip(client_indexes, results):
            sample_number = self.train_data_local_num_dict[client_idx]
            w_locals.append(
                (sample_number, _to_device(result["gmm"], self.device))
            )
        logging.info(
            "Client primary phase mode=%s clients=%d workers=%d "
            "elapsed_seconds=%.6f",
            self.client_executor.execution_mode,
            len(client_indexes), self.client_executor.worker_count,
            time.perf_counter() - phase_start,
        )
        return w_locals

    def _run_correction_client_updates(self, client_indexes, g_global, f_global):
        phase_start = time.perf_counter()
        use_zeroth_order = self.args.client_optimizer == "fed_zo_eg"
        if self.client_executor is None:
            correction_locals = []
            for client in self.client_list:
                if use_zeroth_order:
                    correction = client.train_zo(
                        copy.deepcopy(g_global), copy.deepcopy(f_global)
                    )
                else:
                    correction = client.train(
                        copy.deepcopy(g_global), copy.deepcopy(f_global)
                    )
                correction_locals.append(
                    (client.get_sample_number(), copy.deepcopy(correction))
                )
            logging.info(
                "Client correction phase mode=sp clients=%d elapsed_seconds=%.6f",
                len(client_indexes), time.perf_counter() - phase_start,
            )
            return correction_locals

        tasks = []
        cpu_g_global = _to_cpu(g_global)
        cpu_f_global = _to_cpu(f_global)
        for client_idx in client_indexes:
            tasks.append(
                {
                    "phase": "correction",
                    "use_zeroth_order": use_zeroth_order,
                    "client_idx": int(client_idx),
                    "train_data": self._materialize_client_data(
                        self.train_data_local_dict[client_idx]
                    ),
                    "sample_number": self.train_data_local_num_dict[client_idx],
                    "g_global": cpu_g_global,
                    "f_global": cpu_f_global,
                }
            )
        results = self.client_executor.run(tasks)
        correction_locals = [
            (
                self.train_data_local_num_dict[client_idx],
                _to_device(result["gmm"], self.device),
            )
            for client_idx, result in zip(client_indexes, results)
        ]
        logging.info(
            "Client correction phase mode=%s clients=%d workers=%d "
            "elapsed_seconds=%.6f",
            self.client_executor.execution_mode,
            len(client_indexes), self.client_executor.worker_count,
            time.perf_counter() - phase_start,
        )
        return correction_locals

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        # numpy.random.seed(0)
        # staggler_ids = random.sample(range(10), 5)
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            # additional_epochs = 0
            # if client_idx in staggler_ids:
            #     additional_epochs=10
            c = Client(
                client_idx,
                list(train_data_local_dict[client_idx])[0],
                list(test_data_local_dict[client_idx])[0],
                # train_data_local_dict[client_idx][0],
                # test_data_local_dict[client_idx][0],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        # logging.info("self.model_trainer = {}".format(self.model_trainer))
        # print("Round"+" "+"mse")
        g_global = self.model_trainer.get_g_model_params()
        f_global = self.model_trainer.get_f_model_params()
        reg_global = None if self.skip_auxiliary_regression else self.model_trainer.get_model_params()
        fedAvg=[]
        # mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        # mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        # mlops.log_round_info(self.args.comm_round, -1)
        current_no_progress = 0
        
        start_round = 0
        if hasattr(self.args, 'resume') and self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                logging.info(f"=> loading checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume, map_location=self.device)
                start_round = checkpoint['round'] + 1
                self.model_trainer.set_g_model_params(checkpoint['g_state_dict'])
                self.model_trainer.set_f_model_params(checkpoint['f_state_dict'])
                
                # Initialize self.g and self.f which are normally set in eval_global_model
                self.g = self.model_trainer.g
                self.f = self.model_trainer.f
                
                logging.info(f"=> loaded checkpoint '{self.args.resume}' (round {checkpoint['round']})")
                
                # Update local globals for the first round of resumed training
                g_global = self.model_trainer.get_g_model_params()
                f_global = self.model_trainer.get_f_model_params()
            else:
                logging.error(f"=> no checkpoint found at '{self.args.resume}'")

        training_profile_wall = time.perf_counter()
        training_profile_cpu = time.process_time()
        for round_idx in range(start_round, self.args.comm_round):
            round_profile_wall = time.perf_counter()
            round_profile_cpu = time.process_time()

            # logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            w_locals_reg = [] if not self.skip_auxiliary_regression else None
            w_locals_prev = []
            # obj_sum=[]
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            with self.profiler.span("client_sampling", round_idx=round_idx):
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )
            # logging.info("client_indexes = " + str(client_indexes))
            with self.profiler.span("record_stochastic_batching", round_idx=round_idx):
                self._record_stochastic_batching(round_idx, client_indexes)

            if self.client_executor is None:
                for idx, client in enumerate(self.client_list):
                    # update dataset
                    client_idx = client_indexes[idx]
                    with self.profiler.span("client_update_dataset", round_idx=round_idx, client_id=client_idx):
                        client.update_local_dataset(
                            client_idx,
                            self.train_data_local_dict[client_idx],
                            self.test_data_local_dict[client_idx],
                            self.train_data_local_num_dict[client_idx],
                        )
                    # mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                    # w = client.train(copy.deepcopy(g_global), copy.deepcopy(f_global),copy.deepcopy(w_locals_prev))
                    # optimizer_g = CustomSGD(self.model_trainer.g.parameters(), lr=0.3)
                    # optimizer_f = CustomSGD(self.model_trainer.f.parameters(), lr=0.3)
                    # scheduler_g = CosineAnnealingLR(optimizer_g, T_max=6000)
                    # scheduler_f = CosineAnnealingLR(optimizer_f, T_max=6000)
                    with self.profiler.span("client_train_gmm", round_idx=round_idx, client_id=client_idx):
                        w = client.train(g_global, f_global)

                    # w = client.train()
                    # w=client.train(copy.deepcopy(g_global),copy.deepcopy(f_global))
                    # t=[w[0],w[1]]
                    w_reg = None
                    if self.skip_auxiliary_regression:
                        self.profiler.record(
                            "client_train_reg_skipped",
                            round_idx=round_idx,
                            client_id=client_idx,
                            detail="auxiliary_regression=false",
                        )
                    else:
                        with self.profiler.span("client_train_reg", round_idx=round_idx, client_id=client_idx):
                            w_reg = client.train_reg(reg_global)
                    # mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                    # self.logging.info("local weights = " + str(w))
                    with self.profiler.span("client_collect_state", round_idx=round_idx, client_id=client_idx):
                        w_locals.append((client.get_sample_number(), w))
                        if not self.skip_auxiliary_regression:
                            w_locals_reg.append((client.get_sample_number(), w_reg))
                    # w_locals_prev = t
            else:
                # Parallel client execution (multi-GPU / single-GPU streams).
                # auxiliary_regression is guaranteed off here -- enforced
                # when the executor was constructed -- so there is no reg
                # model to dispatch alongside g/f.
                with self.profiler.span("client_train_gmm_parallel", round_idx=round_idx):
                    w_locals = self._run_primary_client_updates(
                        client_indexes, g_global, f_global
                    )
            with self.profiler.span("record_aggregation_weights", round_idx=round_idx):
                self._record_aggregation_weights(round_idx, client_indexes, w_locals)
            # update global weights
            with self.profiler.span("aggregate_gmm", round_idx=round_idx):
                w_agg = self._aggregate(w_locals)

            with self.profiler.span("server_update_gmm", round_idx=round_idx):
                if self.args.client_optimizer in ("fed_eg", "fed_zo_eg"):
                    # Phase 1 (w_locals/w_agg above) evaluated the local
                    # SGD/FedAvg map at z_t. Build the server predictor from
                    # the aggregated displacement.
                    g_base = self.model_trainer.get_g_model_params()
                    f_base = self.model_trainer.get_f_model_params()
                    predictor_lr = self.args.eg_predictor_server_lr
                    corrector_lr = self.args.eg_corrector_server_lr
                    if predictor_lr is None:
                        predictor_lr = self.server_learning_rate
                    if corrector_lr is None:
                        corrector_lr = self.server_learning_rate

                    predictor_delta_g = {
                        k: w_agg[0][k] - g_base[k] for k in g_base.keys()
                    }
                    predictor_delta_f = {
                        k: w_agg[1][k] - f_base[k] for k in f_base.keys()
                    }
                    g_lookahead = {
                        k: g_base[k] + predictor_lr * predictor_delta_g[k]
                        for k in g_base.keys()
                    }
                    f_lookahead = {
                        k: f_base[k] + predictor_lr * predictor_delta_f[k]
                        for k in f_base.keys()
                    }

                    # Phase 2 uses the same sampled clients and local
                    # datasets, now initialized at the look-ahead model.
                    correction_locals = self._run_correction_client_updates(
                        client_indexes, g_lookahead, f_lookahead
                    )

                    correction_agg = self._aggregate(correction_locals)
                    correction_delta_g = {
                        k: correction_agg[0][k] - g_lookahead[k]
                        for k in g_base.keys()
                    }
                    correction_delta_f = {
                        k: correction_agg[1][k] - f_lookahead[k]
                        for k in f_base.keys()
                    }

                    # The EG corrector is anchored at z_t, not at z_bar.
                    g_new = {
                        k: g_base[k] + corrector_lr * correction_delta_g[k]
                        for k in g_base.keys()
                    }
                    f_new = {
                        k: f_base[k] + corrector_lr * correction_delta_f[k]
                        for k in f_base.keys()
                    }
                    w_global = [g_new, f_new]
                elif self.args.client_optimizer == "ogda":
                    server_lr = self.server_learning_rate

                    # Current weights (theta_t)
                    g_old = self.model_trainer.get_g_model_params()
                    f_old = self.model_trainer.get_f_model_params()

                    # Current updates (delta_t = w_agg - theta_t)
                    delta_g = {k: w_agg[0][k] - g_old[k] for k in g_old.keys()}
                    delta_f = {k: w_agg[1][k] - f_old[k] for k in f_old.keys()}

                    if round_idx == start_round or self.delta_g_prev is None:
                        # Round 1: theta_{t+1} = theta_t + beta * delta_t
                        g_new = {k: g_old[k] + server_lr * delta_g[k] for k in g_old.keys()}
                        f_new = {k: f_old[k] + server_lr * delta_f[k] for k in f_old.keys()}
                    else:
                        # Round t > 1: Optimistic Update
                        # theta_{t+1} = theta_t + beta * (2*delta_t - delta_{t-1})
                        g_new = {k: g_old[k] + server_lr * (2.0 * delta_g[k] - self.delta_g_prev[k]) for k in g_old.keys()}
                        f_new = {k: f_old[k] + server_lr * (2.0 * delta_f[k] - self.delta_f_prev[k]) for k in f_old.keys()}

                    # Store deltas for the next round
                    self.delta_g_prev = delta_g
                    self.delta_f_prev = delta_f
                    w_global = [g_new, f_new]
                else:
                    # w_global = w_agg
                    server_lr = self.server_learning_rate

                    # Current weights (theta_t)
                    g_old = self.model_trainer.get_g_model_params()
                    f_old = self.model_trainer.get_f_model_params()

                    # Current updates (delta_t = w_agg - theta_t)
                    delta_g = {k: w_agg[0][k] - g_old[k] for k in g_old.keys()}
                    delta_f = {k: w_agg[1][k] - f_old[k] for k in f_old.keys()}


                    #theta_{t+1} = theta_t + beta * delta_t
                    g_new = {k: g_old[k] + server_lr * delta_g[k] for k in g_old.keys()}
                    f_new = {k: f_old[k] + server_lr * delta_f[k] for k in f_old.keys()}

                    w_global = [g_new, f_new]

            w_global_reg = None
            if self.skip_auxiliary_regression:
                self.profiler.record(
                    "aggregate_reg_skipped",
                    round_idx=round_idx,
                    detail="auxiliary_regression=false",
                )
            else:
                with self.profiler.span("aggregate_reg", round_idx=round_idx):
                    w_global_reg = self._aggregate_reg(w_locals_reg)
            with self.profiler.span("set_global_params", round_idx=round_idx):
                self.model_trainer.set_g_model_params(w_global[0])
                self.model_trainer.set_f_model_params(w_global[1])
                if not self.skip_auxiliary_regression:
                    self.model_trainer.set_model_params(w_global_reg)
                g_global = self.model_trainer.get_g_model_params()
                f_global = self.model_trainer.get_f_model_params()
                if not self.skip_auxiliary_regression:
                    reg_global = self.model_trainer.get_model_params()
            # mlops.event("agg", event_started=False, event_value=str(round_idx))

            # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            with self.profiler.span("eval_global_model", round_idx=round_idx):
                eval_result = self.eval_global_model()
            mse = eval_result["train_mse"]
            obj_train = eval_result["gmm_train_objective"]
            obj_dev = eval_result["gmm_val_objective"]
            curr_eval = eval_result["gmm_eval"]
            max_recent_eval = eval_result["max_recent_gmm_eval"]
            f_of_z_train = eval_result["f_of_z_train"]
            f_of_z_dev = eval_result["f_of_z_dev"]
            state_finite = (
                state_is_finite(self.model_trainer.get_g_model_params())
                and state_is_finite(self.model_trainer.get_f_model_params())
            )
            metrics_finite = metric_values_are_finite(
                eval_result[key]
                for key in (
                    "train_mse",
                    "val_mse",
                    "primary_val_mse",
                    "equal_client_val_mse",
                    "sample_weighted_val_mse",
                    "train_moment_violation",
                    "val_moment_violation",
                    "equal_client_val_moment_violation",
                    "sample_weighted_val_moment_violation",
                    "gmm_train_objective",
                    "gmm_val_objective",
                    "gmm_eval",
                )
            )
            finite = state_finite and metrics_finite
            diverged = not finite
            if diverged and self.nonfinite_first_round is None:
                self.nonfinite_first_round = round_idx
            self.diverged = self.diverged or diverged
            mse_row = {
                "round": round_idx,
                "train_mse": eval_result["train_mse"],
                "val_mse": eval_result["val_mse"],
                "primary_val_mse": eval_result["primary_val_mse"],
                "equal_client_val_mse": eval_result["equal_client_val_mse"],
                "train_moment_violation": eval_result["train_moment_violation"],
                "val_moment_violation": eval_result["val_moment_violation"],
                "gmm_train_objective": obj_train,
                "gmm_val_objective": obj_dev,
                "gmm_eval": curr_eval,
                "finite": finite,
                "diverged": diverged,
            }
            self.mse_rows.append(mse_row)
            with self.profiler.span("write_mse_by_round", round_idx=round_idx, detail=f"rows={len(self.mse_rows)}"):
                if self.append_round_csv:
                    append_mse_by_round(self.run_dir, mse_row)
                else:
                    write_mse_by_round(self.run_dir, self.mse_rows)
            if self.log_test_mse_by_round:
                with self.profiler.span("test_mse_by_round_eval", round_idx=round_idx):
                    test_mse = structural_mse(self.model_trainer.g, self.test_global)
                test_finite = math.isfinite(float(test_mse))
                test_mse_row = {
                    "round": round_idx,
                    "test_mse": test_mse,
                    "finite": test_finite,
                    "diverged": diverged or not test_finite,
                }
                self.test_mse_rows.append(test_mse_row)
                with self.profiler.span("write_test_mse_by_round", round_idx=round_idx, detail=f"rows={len(self.test_mse_rows)}"):
                    if self.append_round_csv:
                        append_test_mse_by_round(self.run_dir, test_mse_row)
                    else:
                        write_test_mse_by_round(self.run_dir, self.test_mse_rows)
            if getattr(self.args, "enable_legacy_outputs", True):
                with self.profiler.span("legacy_csv_write", round_idx=round_idx):
                    log_results_to_csv(f"csv/{self.args.client_optimizer}_{self.args.dataset}newtrial.csv", round_idx, mse)
            # Primary checkpoint selector: equal-client validation structural
            # MSE for eICU (protocol_v1.md S5.1), which is identical to the
            # legacy pooled val_mse for every other dataset (primary_val_mse
            # == val_mse whenever client_id is unavailable).
            if eval_result["primary_val_mse"] < self.best_validation_mse:
                with self.profiler.span("best_validation_checkpoint", round_idx=round_idx):
                    self.best_validation_mse = eval_result["primary_val_mse"]
                    self.best_validation_round = round_idx
                    self.best_state = self._capture_training_state()
                    self.best_g_state = copy.deepcopy(self.best_state["g_state_dict"])
                    self.best_f_state = copy.deepcopy(self.best_state["f_state_dict"])
                    self.best_reg_state = copy.deepcopy(self.best_state["reg_state_dict"])
                    self._save_checkpoint(
                        os.path.join(self.run_dir, "checkpoints", "best_validation.pt"),
                        round_idx,
                        self.best_state,
                        eval_result,
                        "best_validation",
                    )
            _primary_val_moment_violation = (
                eval_result["equal_client_val_moment_violation"]
                if eval_result["equal_client_val_moment_violation"] is not None
                else eval_result["val_moment_violation"]
            )
            if _primary_val_moment_violation < self.best_moment_violation:
                with self.profiler.span("best_moment_violation_checkpoint", round_idx=round_idx):
                    self.best_moment_violation = _primary_val_moment_violation
                    self.best_moment_violation_round = round_idx
                    self.best_moment_state = self._capture_training_state()
                    self._save_checkpoint(
                        os.path.join(self.run_dir, "checkpoints", "best_moment_violation.pt"),
                        round_idx,
                        self.best_moment_state,
                        eval_result,
                        "best_moment_violation",
                    )

            # Save periodic checkpoints inside the unique run directory.
            if self.periodic_checkpoint_interval > 0 and round_idx % self.periodic_checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.run_dir, "checkpoints", f"round_{round_idx}.pt")
                with self.profiler.span("periodic_checkpoint", round_idx=round_idx):
                    self._save_checkpoint(
                        checkpoint_path,
                        round_idx,
                        self._capture_training_state(),
                        eval_result,
                        "periodic",
                    )
                if getattr(self.args, "enable_legacy_outputs", True):
                    with self.profiler.span("legacy_checkpoint", round_idx=round_idx):
                        legacy_checkpoint_path = f"checkpoints/{self.args.client_optimizer}_{self.args.dataset}_round_{round_idx}.pt"
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save({
                            "round": round_idx,
                            "g_state_dict": self.g.state_dict(),
                            "f_state_dict": self.f.state_dict(),
                            "mse": mse,
                        }, legacy_checkpoint_path)
            # wandb.log({"round":round_idx,"MSE" :mse})
            # logging.info(f"{round_idx}: {mse:.4f}")
            # print(round_idx,end=" ")
            # print(mse)
            # fedAvg.append(mse)
            if round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    pass
                
                if self.args.video_plotter and round_idx % self.args.print_freq == 0:
                    frame = self.video_plotter.get_new_frame("iter = %d" % round_idx)

                    self.f = self.f.eval()
                    self.g = self.g.eval()

                    # plot f(z)
                    frame.add_plot(PlotElement(
                        self.train_global.w.cpu().numpy(), f_of_z_train.numpy(),
                        "estimated f(z)", normalize=True))

                    # plot g(x)
                    g_of_x_plot = self.epsilon_train_history[-1] + self.train_global.y.cpu()
                    frame.add_plot(PlotElement(self.train_global.w.cpu().numpy(), g_of_x_plot.numpy(),
                                            "fitted g(x)"))

                    self.f = self.f.train()
                    self.g = self.g.train()
                    
                # if round_idx % self.args.print_freq == 0 and self.args.verbose:
                #     mean_eval = numpy.mean(self.eval_history[-self.args.print_freq_mul:])
                #     print("iteration %d, dev-MSE=%f, train-loss=%f,"
                #         " dev-loss=%f, mean-recent-eval=%f"
                #         % (round_idx, mse, obj_train, obj_dev, mean_eval))
                    # wandb.log({"round": round_idx, "MSE": mse, "Train-Loss": obj_train, "Test-Loss": obj_dev, "Objective": mean_eval})

            self.profiler.record(
                "round_total",
                time.perf_counter() - round_profile_wall,
                process_cpu_seconds=time.process_time() - round_profile_cpu,
                round_idx=round_idx,
            )

            # check stopping conditions if we are past burn-in
            if round_idx % self.args.eval_freq == 0 and round_idx >= self.args.burn_in:
                if curr_eval > max_recent_eval:
                    current_no_progress = 0
                else:
                    current_no_progress += 1

                if current_no_progress >= self.args.max_no_progress:
                    break
        if self.client_executor is not None:
            self.client_executor.close()
            self.client_executor = None
        self.profiler.record(
            "training_total",
            time.perf_counter() - training_profile_wall,
            process_cpu_seconds=time.process_time() - training_profile_cpu,
        )
        if self.batching_summary_rows:
            write_batching_summary(self.run_dir, self.batching_summary_rows)
        # plot relationship between MSE and eval
        # if self.args.video_plotter:
        #     plt.figure()
        #     data = pandas.DataFrame({"eval": self.eval_list, "mse": self.mse_list})
        #     data.plot.scatter(x="eval", y="mse")
        #     plt.savefig("eval_mse.png")
        finalization_profile_wall = time.perf_counter()
        finalization_profile_cpu = time.process_time()
        max_i = max(range(len(self.eval_history)), key=lambda i_: self.eval_history[i_])
        if self.args.verbose:
            # print("best iteration:", self.args.eval_freq * max_i)
            pass
            # mlops.log_round_info(self.args.comm_round, round_idx)
        best_gmm_eval_round = self.args.eval_freq * max_i
        final_round = round_idx if "round_idx" in locals() else -1
        with self.profiler.span("final_capture_state"):
            self.final_state = self._capture_training_state()
            self.final_g_state = copy.deepcopy(self.final_state["g_state_dict"])
            self.final_f_state = copy.deepcopy(self.final_state["f_state_dict"])
            self.final_reg_state = copy.deepcopy(self.final_state["reg_state_dict"])
            self._load_training_state(self.final_state)
        val_client_id = getattr(self.val_global, "client_id", None)
        test_client_id = getattr(self.test_global, "client_id", None)
        with self.profiler.span("final_validation_test_eval"):
            final_validation_mse = structural_mse(self.model_trainer.g, self.val_global)
            final_test_mse = structural_mse(self.model_trainer.g, self.test_global)
            final_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            final_train_mse = structural_mse(self.model_trainer.g, self.train_global)
            final_val_g_prediction = predictions_for_split(self.model_trainer.g, self.val_global)
            self.model_trainer.f.eval()
            with torch.no_grad():
                final_val_f_prediction = self.model_trainer.f(self.val_global.z)
            self.model_trainer.f.train()
            final_moment_violation = moment_violation(
                final_val_f_prediction, self._val_y_cpu, final_val_g_prediction
            )
            # protocol_v1.md S5.2/metric_policy.md: report equal-client and
            # sample-weighted structural MSE for the final iterate too, not
            # only the best-validation checkpoint. Gated on client_id exactly
            # like eval_global_model, so non-eICU behavior is untouched.
            equal_client_final_validation_structural_mse = None
            sample_weighted_final_validation_structural_mse = None
            primary_final_validation_mse = final_validation_mse
            if val_client_id is not None:
                (
                    equal_client_final_validation_structural_mse,
                    sample_weighted_final_validation_structural_mse,
                    _,
                ) = equal_client_structural_mse(final_val_g_prediction, self._val_g_cpu, val_client_id)
                primary_final_validation_mse = equal_client_final_validation_structural_mse
            equal_client_final_test_structural_mse = None
            sample_weighted_final_test_structural_mse = None
            primary_final_test_mse = final_test_mse
            if test_client_id is not None:
                (
                    equal_client_final_test_structural_mse,
                    sample_weighted_final_test_structural_mse,
                    _,
                ) = equal_client_structural_mse(final_prediction, self._test_g_cpu, test_client_id)
                primary_final_test_mse = equal_client_final_test_structural_mse
        with self.profiler.span("final_checkpoint"):
            self._save_checkpoint(
                os.path.join(self.run_dir, "checkpoints", "final.pt"),
                final_round,
                self.final_state,
                {
                    "train_mse": final_train_mse,
                    "val_mse": final_validation_mse,
                    "test_mse": final_test_mse,
                },
                "final",
            )
        if self.best_state is None:
            self.best_validation_round = final_round
            self.best_validation_mse = primary_final_validation_mse
            self.best_state = copy.deepcopy(self.final_state)
            self.best_g_state = copy.deepcopy(self.final_g_state)
            self.best_f_state = copy.deepcopy(self.final_f_state)
            self.best_reg_state = copy.deepcopy(self.final_reg_state)
        if self.best_moment_state is None:
            self.best_moment_violation_round = final_round
            self.best_moment_violation = final_moment_violation
            self.best_moment_state = copy.deepcopy(self.final_state)

        with self.profiler.span("best_validation_test_eval"):
            self._load_training_state(self.best_state)
            best_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            best_test_mse = structural_mse_from_predictions(best_prediction, self._test_g_cpu)
            best_val_g_prediction = predictions_for_split(self.model_trainer.g, self.val_global)
            self.model_trainer.f.eval()
            with torch.no_grad():
                best_val_f_prediction = self.model_trainer.f(self.val_global.z)
                best_test_f_prediction = self.model_trainer.f(self.test_global.z)
            self.model_trainer.f.train()

            # protocol_v1.md S5.1: for eICU the compatibility field
            # test_mse_at_best_validation must itself carry equal-client
            # semantics (not a sample-weighted/pooled value under that name)
            # -- metric_policy.md's selection firewall explicitly forbids
            # substituting one for the other.
            equal_client_test_mse_at_best_validation = None
            sample_weighted_test_mse_at_best_validation = None
            per_client_test_mse_at_best_validation = None
            primary_test_mse_at_best_validation = best_test_mse
            if test_client_id is not None:
                (
                    equal_client_test_mse_at_best_validation,
                    sample_weighted_test_mse_at_best_validation,
                    per_client_test_mse_at_best_validation,
                ) = equal_client_structural_mse(best_prediction, self._test_g_cpu, test_client_id)
                primary_test_mse_at_best_validation = equal_client_test_mse_at_best_validation

            equal_client_best_validation_structural_mse = None
            sample_weighted_validation_mse_at_best_validation = None
            equal_client_validation_moment_violation_at_best_validation = None
            sample_weighted_validation_moment_violation_at_best_validation = None
            if val_client_id is not None:
                (
                    equal_client_best_validation_structural_mse,
                    sample_weighted_validation_mse_at_best_validation,
                    _,
                ) = equal_client_structural_mse(best_val_g_prediction, self._val_g_cpu, val_client_id)
                (
                    equal_client_validation_moment_violation_at_best_validation,
                    sample_weighted_validation_moment_violation_at_best_validation,
                    _,
                ) = equal_client_moment_violation(
                    best_val_f_prediction, self._val_y_cpu, best_val_g_prediction, val_client_id
                )

            equal_client_test_moment_violation_at_best_validation = None
            sample_weighted_test_moment_violation_at_best_validation = None
            per_client_test_moment_violation_at_best_validation = None
            if test_client_id is not None:
                (
                    equal_client_test_moment_violation_at_best_validation,
                    sample_weighted_test_moment_violation_at_best_validation,
                    per_client_test_moment_violation_at_best_validation,
                ) = equal_client_moment_violation(
                    best_test_f_prediction, self._test_y_cpu, best_prediction, test_client_id
                )

            per_client_metrics_artifact = None
            if test_client_id is not None:
                per_client_metrics_artifact = write_per_client_metrics(
                    self.run_dir,
                    per_client_test_mse_at_best_validation,
                    per_client_test_moment_violation_at_best_validation,
                )
        print("MSE on test ------------------------------>>>>>>>>>>>>>>>>>>", best_test_mse)

        with self.profiler.span("best_moment_violation_test_eval"):
            self._load_training_state(self.best_moment_state)
            best_moment_prediction = predictions_for_split(self.model_trainer.g, self.test_global)
            test_mse_at_best_moment_violation = structural_mse_from_predictions(
                best_moment_prediction, self._test_g_cpu
            )
        # Restore the val-MSE-selected state as "the" checkpoint used for
        # save_predictions_npz below, matching the field names' existing meaning.
        self._load_training_state(self.best_state)

        with self.profiler.span("save_predictions_npz"):
            save_predictions_npz(
                self.run_dir,
                self.test_global,
                best_prediction,
                final_prediction,
                self.effective_config,
            )

        # Stability diagnostics (metric_policy.md "Stability metrics"):
        # final-minus-best gaps and the spread of the primary (equal-client
        # when available, else pooled) validation curve over its last 50
        # recorded rounds.
        final_vs_best_validation_gap = primary_final_validation_mse - self.best_validation_mse
        final_vs_best_test_gap = primary_final_test_mse - primary_test_mse_at_best_validation
        recent_primary_val_mse = [
            row["primary_val_mse"] for row in self.mse_rows[-50:] if row.get("primary_val_mse") is not None
        ]
        if recent_primary_val_mse:
            last_50_validation_mse_std = float(numpy.std(recent_primary_val_mse))
            last_50_validation_mse_range = float(max(recent_primary_val_mse) - min(recent_primary_val_mse))
        else:
            last_50_validation_mse_std = None
            last_50_validation_mse_range = None

        metrics = {
            "run_id": self.effective_config.get("run_id", ""),
            "method": self.effective_config.get("variant", ""),
            "seed": self.effective_config.get("random_seed", 0),
            "scenario_seed": self.effective_config.get("scenario_seed", 0),
            "optimizer_seed": self.effective_config.get("optimizer_seed", 0),
            "best_validation_round": self.best_validation_round,
            "best_validation_mse": self.best_validation_mse,
            "test_mse_at_best_validation": primary_test_mse_at_best_validation,
            "best_moment_violation_round": self.best_moment_violation_round,
            "best_moment_violation": self.best_moment_violation,
            "test_mse_at_best_moment_violation": test_mse_at_best_moment_violation,
            "final_moment_violation": final_moment_violation,
            "final_validation_mse": primary_final_validation_mse,
            "final_test_mse": primary_final_test_mse,
            "best_gmm_eval_round": best_gmm_eval_round,
            "best_gmm_eval": self.eval_history[max_i],
            "diverged": self.diverged,
            "nonfinite_first_round": self.nonfinite_first_round,
            "run_status": "completed",
            "failure_reason": None,
            "rounds_completed": len(self.mse_rows),
            "runtime_seconds": now_seconds() - self.start_time,
            "test_mse_logged_by_round": self.log_test_mse_by_round,
            "test_mse_used_for_selection": False,
            "selection_metric_source": "validation",
            "selection_metric": self.effective_config.get(
                "primary_selection_metric", "equal_client_validation_mse"
            ),
            "selection_source": self.effective_config.get(
                "selection_source", "validation_only"
            ),
            "selected_round": self.best_validation_round,
            "test_mse_reported_after_selection": True,
            "is_primary": str(getattr(self.args, "campaign_role", "")) != "aggregation_ablation",
            "alignment_label": self.effective_config.get("alignment_label", ""),
            "skip_model_selection": self.skip_model_selection,
            "skip_gmm_eval": self.skip_gmm_eval,
            "gmm_eval_proxy": self.gmm_eval_proxy,
            "auxiliary_regression": not self.skip_auxiliary_regression,
            "auxiliary_regression_epochs": int(getattr(self.args, "auxiliary_regression_epochs", 0)),
            "append_round_csv": self.append_round_csv,
            "periodic_checkpoint_interval": self.periodic_checkpoint_interval,
            "aggregation_weighting": self.aggregation_weighting,
            "objective_mode": self.objective_mode,
            "scenario_checksum": self.effective_config.get("scenario_checksum", ""),
            "config_checksum": config_checksum(self.effective_config),
            "per_client_metrics_artifact": per_client_metrics_artifact,
            "curve_artifact": "mse_by_round.csv",
            "best_checkpoint_artifact": os.path.join("checkpoints", "best_validation.pt"),
            "final_checkpoint_artifact": os.path.join("checkpoints", "final.pt"),
            "effective_config_artifact": "effective_config.json",
            "equal_client_best_validation_structural_mse": equal_client_best_validation_structural_mse,
            "sample_weighted_validation_mse_at_best_validation": sample_weighted_validation_mse_at_best_validation,
            "equal_client_final_validation_structural_mse": equal_client_final_validation_structural_mse,
            "sample_weighted_final_validation_structural_mse": sample_weighted_final_validation_structural_mse,
            "equal_client_test_mse_at_best_validation": equal_client_test_mse_at_best_validation,
            "sample_weighted_test_mse_at_best_validation": sample_weighted_test_mse_at_best_validation,
            "equal_client_final_test_structural_mse": equal_client_final_test_structural_mse,
            "sample_weighted_final_test_structural_mse": sample_weighted_final_test_structural_mse,
            "equal_client_validation_moment_violation_at_best_validation": equal_client_validation_moment_violation_at_best_validation,
            "sample_weighted_validation_moment_violation_at_best_validation": sample_weighted_validation_moment_violation_at_best_validation,
            "equal_client_test_moment_violation_at_best_validation": equal_client_test_moment_violation_at_best_validation,
            "sample_weighted_test_moment_violation_at_best_validation": sample_weighted_test_moment_violation_at_best_validation,
            # Filled in post-hoc by analyze_eicu_study_a_checkpoint.py, which
            # is the only place that knows how to build D=0/1 counterfactual
            # rows for a given Study A scenario (see metric_policy.md
            # "ATE error and individual-effect MAE" -- these are not
            # substitutable and both require the scenario's true_effect
            # array, unavailable to the dataset-agnostic trainer here).
            "equal_client_absolute_ate_error_at_best_validation": None,
            "sample_weighted_absolute_ate_error_at_best_validation": None,
            "equal_client_individual_effect_mae_at_best_validation": None,
            "sample_weighted_individual_effect_mae_at_best_validation": None,
            "final_vs_best_validation_gap": final_vs_best_validation_gap,
            "final_vs_best_test_gap": final_vs_best_test_gap,
            "last_50_validation_mse_std": last_50_validation_mse_std,
            "last_50_validation_mse_range": last_50_validation_mse_range,
        }
        with self.profiler.span("write_metrics"):
            write_metrics(self.run_dir, metrics)

        if getattr(self.args, "enable_legacy_outputs", True) or getattr(self.args, "enable_legacy_plot", False):
            with self.profiler.span("legacy_plot"):
                best_pred_np = best_prediction.detach().cpu().numpy()
                g_true = self.test_global.g.detach().cpu().numpy()
                x_sort, best_pred_sort, g_true_sort = prepare_curve_data(
                    self.test_global, best_pred_np, g_true
                )
                pred_plot = PlotElement(x_sort, best_pred_sort, "FedDeepGMM-SGDA")
                true_plot = PlotElement(x_sort, g_true_sort, "Actual Causal Effect")
                plot_dir = "plots" if getattr(self.args, "enable_legacy_outputs", True) else os.path.join(self.run_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                fig, ax = plt.subplots()
                ax = true_plot.plot(ax=ax, save_path=os.path.join(plot_dir, f'aaaa_{self.args.run_name}_.png'))
                ax = pred_plot.plot(ax=ax, save_path=os.path.join(plot_dir, f'aaaa_{self.args.run_name}_.png'))
        # ax = reg_NN_plot.plot(ax=ax, save_path=f'plots/aaaacomparison_{self.args.run_name}_.png')
        self.profiler.record(
            "finalization",
            time.perf_counter() - finalization_profile_wall,
            process_cpu_seconds=time.process_time() - finalization_profile_cpu,
        )
        
        # mlops.log_training_finished_status()
        # mlops.log_aggregation_finished_status()
        
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        client_indexes = client_indices_for_round(
            getattr(self.args, "random_seed", 0),
            round_idx,
            client_num_in_total,
            client_num_per_round,
        )
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _record_stochastic_batching(self, round_idx, client_indexes):
        if not bool(getattr(self.args, "require_multibatch_stochastic", False)):
            return
        batch_size = int(getattr(self.args, "batch_size", 0))
        if batch_size <= 0:
            raise ValueError("require_multibatch_stochastic requires a positive batch_size")
        variant = self.effective_config.get("variant", "unknown")
        for client_idx in client_indexes:
            local_loader = self.train_data_local_dict[client_idx]
            sample_count = int(self.train_data_local_num_dict[client_idx])
            try:
                num_batches = int(len(local_loader))
            except TypeError:
                num_batches = int(math.ceil(sample_count * 1.0 / batch_size))
            first_actual_batch_size = min(sample_count, batch_size) if sample_count > 0 else 0
            row = {
                "variant": variant,
                "round": int(round_idx),
                "client_id": int(client_idx),
                "sample_count": sample_count,
                "configured_batch_size": batch_size,
                "num_batches": num_batches,
                "first_actual_batch_size": first_actual_batch_size,
            }
            self.batching_summary_rows.append(row)
            if num_batches < 2:
                write_batching_summary(self.run_dir, self.batching_summary_rows)
                raise ValueError(
                    "Stochastic smoke multibatch guard failed: "
                    f"variant={variant}, round={round_idx}, client_id={client_idx}, "
                    f"sample_count={sample_count}, configured_batch_size={batch_size}, "
                    f"num_batches={num_batches}"
                )
        # The file is written once after the round loop, not here. Rewriting the
        # whole accumulated list every round is quadratic in comm_round: the list
        # grows by client_num_per_round entries per round, so a 2000-round run
        # serialized ~358M rows to produce a 64MB artifact. The guard above still
        # flushes on failure, so a violating run keeps its forensic record.

    def _record_aggregation_weights(self, round_idx, client_indexes, w_locals):
        """Persist the resolved aggregation mode and the actual per-client
        weights used this round, so ``uniform_clients`` vs ``sample_size`` is
        independently auditable from the run directory rather than only from
        ``effective_config.json``'s static mode string. One row per round,
        matching every other ``*_by_round.csv`` artifact's granularity.
        """
        weights = self._client_weights(w_locals)
        client_weights = {
            int(client_idx): float(weight)
            for client_idx, weight in zip(client_indexes, weights)
        }
        row = {
            "round": int(round_idx),
            "aggregation_weighting": self.aggregation_weighting,
            "k_participating": len(client_weights),
            "weight_sum": float(sum(client_weights.values())),
            "client_weights_json": json.dumps(client_weights, sort_keys=True),
        }
        self.aggregation_weight_rows.append(row)
        if self.append_round_csv:
            append_aggregation_weights_by_round(self.run_dir, row)
        else:
            write_aggregation_weights_by_round(self.run_dir, self.aggregation_weight_rows)

    def _capture_training_state(self):
        return {
            "g_state_dict": copy.deepcopy(self.model_trainer.get_g_model_params()),
            "f_state_dict": copy.deepcopy(self.model_trainer.get_f_model_params()),
            "reg_state_dict": (
                None
                if self.skip_auxiliary_regression
                else copy.deepcopy(self.model_trainer.get_model_params())
            ),
        }

    def _load_training_state(self, state):
        self.model_trainer.set_g_model_params(copy.deepcopy(state["g_state_dict"]))
        self.model_trainer.set_f_model_params(copy.deepcopy(state["f_state_dict"]))
        if state.get("reg_state_dict") is not None:
            self.model_trainer.set_model_params(copy.deepcopy(state["reg_state_dict"]))
        self.g = self.model_trainer.g
        self.f = self.model_trainer.f
        self.reg_model = self.model_trainer.reg_model

    def _aggregate_t(self,obj_sum):
        total_sum=0
        for i in obj_sum:
            total_sum+=i
        return total_sum/10
    def _client_weights(self, w_locals):
        """Per-client weights for this round, per ``self.aggregation_weighting``.

        Shared by ``_aggregate`` and ``_aggregate_reg`` so FedGDA's direct
        update and FedOGDA's pseudogradient/delta (which is derived as
        ``weighted_average(w_locals) - theta_t``, see ``train()``) always use
        the identical, consistent weighting for both g and f.
        """
        sample_counts = [num for num, _ in w_locals]
        return compute_client_weights(sample_counts, self.aggregation_weighting)

    def _aggregate_reg(self, w_locals):
        weights = self._client_weights(w_locals)
        state_dicts = [params for _, params in w_locals]
        return weighted_average_state_dicts(state_dicts, weights)

    def _aggregate(self, w_locals):
        weights = self._client_weights(w_locals)
        g_dicts = [gf[0] for _, gf in w_locals]
        f_dicts = [gf[1] for _, gf in w_locals]
        g = weighted_average_state_dicts(g_dicts, weights)
        f = weighted_average_state_dicts(f_dicts, weights)
        return [g, f]

    def _save_checkpoint(self, path, round_idx, state, metrics, checkpoint_type):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "round": round_idx,
            "checkpoint_type": checkpoint_type,
            "state": {
                "g": state["g_state_dict"],
                "f": state["f_state_dict"],
                "regression": state.get("reg_state_dict"),
            },
            "g_state_dict": state["g_state_dict"],
            "f_state_dict": state["f_state_dict"],
            "reg_state_dict": state.get("reg_state_dict"),
            "metrics": metrics,
            "effective_config": self.effective_config,
        }, path)

    def _effective_batch_size(self, num_data):
        if not hasattr(self.args, "batch_size") or self.args.batch_size is None or self.args.batch_size <= 0:
            return num_data
        return self.args.batch_size

    def calc_f_g_obj(self, global_val):
        x = global_val.x
        y = global_val.y
        z = global_val.z
        num_data = x.shape[0]
        batch_size = self._effective_batch_size(num_data)
        num_batch = math.ceil(num_data * 1.0 / batch_size)
        g_of_x_batches = []
        f_of_z_batches = []
        obj_total = 0
        with torch.no_grad():
            for b in range(num_batch):
                batch_idx = slice(b * batch_size, min((b + 1) * batch_size, num_data))
                x_batch = x[batch_idx]
                z_batch = z[batch_idx]
                y_batch = y[batch_idx]
                g_obj, _ = self.model_trainer.game_objective.calc_objective(self.model_trainer.g, self.model_trainer.f, x_batch, z_batch, y_batch)
                g_of_x_batches.append(self.model_trainer.g(x_batch).detach().cpu())
                f_of_z_batches.append(self.model_trainer.f(z_batch).detach().cpu())
                obj_total += float(g_obj.detach().cpu()) * int(x_batch.shape[0]) * 1.0 / num_data
        g_of_x = torch.cat(g_of_x_batches, dim=0)
        f_of_z = torch.cat(f_of_z_batches, dim=0)
        return g_of_x, f_of_z, obj_total
    
        
    def eval_global_model(self):
        # self.model_trainer is never trained directly (each client works on
        # its own copy.deepcopy of it, see _setup_clients), so its
        # game_objective instance never gets set_theta_tilde called on it by
        # any client. calc_f_g_obj below only uses calc_objective for the
        # informational gmm_train_objective/gmm_val_objective fields (the
        # moment-violation metric is computed independently, directly from
        # g_of_x/f_of_z/y), so any theta~ is fine here -- use the current
        # (just-aggregated) global g.
        if hasattr(self.model_trainer.game_objective, "set_theta_tilde"):
            self.model_trainer.game_objective.set_theta_tilde(self.model_trainer.g)
        self.f = self.model_trainer.f.eval()
        self.g = self.model_trainer.g.eval()
        g_of_x_train, f_of_z_train, obj_train = self.calc_f_g_obj(self.train_global)
        g_of_x_dev, f_of_z_dev, obj_dev = self.calc_f_g_obj(self.val_global)
        epsilon_dev = g_of_x_dev - self._val_y_cpu
        epsilon_train = g_of_x_train - self._train_y_cpu
        train_mse = structural_mse_from_predictions(g_of_x_train, self._train_g_cpu)
        val_mse = structural_mse_from_predictions(g_of_x_dev, self._val_g_cpu)
        # Held-out moment violation: g0-free, so this is the one recovery
        # diagnostic that also works on real (Study B) data where val_mse
        # cannot be computed at all.
        train_moment_violation = moment_violation(
            f_of_z_train, self._train_y_cpu, g_of_x_train
        )
        val_moment_violation = moment_violation(f_of_z_dev, self._val_y_cpu, g_of_x_dev)

        # protocol_v1.md S5.1/metric_policy.md: Study A's primary checkpoint
        # selector is equal-client (per-hospital-averaged) validation
        # structural MSE, not this pooled/sample-weighted val_mse. client_id
        # survives on val_global untouched (Dataset.to_tensor/the eicu data
        # loader never wrap or strip it -- see abstract_scenario.py), so it is
        # available here for every eICU run and absent (None) for every other
        # dataset family, which is exactly the gate this block uses: legacy
        # datasets get val_mse unchanged, byte for byte.
        val_client_id = getattr(self.val_global, "client_id", None)
        equal_client_val_mse = None
        sample_weighted_val_mse = None
        per_client_val_mse = None
        equal_client_val_moment_violation = None
        sample_weighted_val_moment_violation = None
        per_client_val_moment_violation = None
        primary_val_mse = val_mse
        if val_client_id is not None:
            equal_client_val_mse, sample_weighted_val_mse, per_client_val_mse = (
                equal_client_structural_mse(g_of_x_dev, self._val_g_cpu, val_client_id)
            )
            (
                equal_client_val_moment_violation,
                sample_weighted_val_moment_violation,
                per_client_val_moment_violation,
            ) = equal_client_moment_violation(
                f_of_z_dev, self._val_y_cpu, g_of_x_dev, val_client_id
            )
            primary_val_mse = equal_client_val_mse
        if self.skip_gmm_eval:
            curr_eval = -float(val_mse)
        else:
            curr_eval = approx_psi_eval(epsilon_dev, self.dev_f_collection,
                                                self.e_dev_tilde)
        self.eval_list.append(curr_eval)
        self.mse_list.append(train_mse)
        if self.eval_history:
            max_recent_eval = max(self.eval_history)
        else:
            max_recent_eval = float("-inf")
        self.eval_history.append(curr_eval)
        if self.args.video_plotter:
            self.epsilon_dev_history.append(epsilon_dev)
            self.epsilon_train_history.append(epsilon_train)
            self.g_state_history.append(copy.deepcopy(self.g.state_dict()))

        self.f = self.f.train()
        self.g = self.g.train()
        return {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "primary_val_mse": primary_val_mse,
            "equal_client_val_mse": equal_client_val_mse,
            "sample_weighted_val_mse": sample_weighted_val_mse,
            "per_client_val_mse": per_client_val_mse,
            "train_moment_violation": train_moment_violation,
            "val_moment_violation": val_moment_violation,
            "equal_client_val_moment_violation": equal_client_val_moment_violation,
            "sample_weighted_val_moment_violation": sample_weighted_val_moment_violation,
            "per_client_val_moment_violation": per_client_val_moment_violation,
            "gmm_train_objective": obj_train,
            "gmm_val_objective": obj_dev,
            "gmm_eval": curr_eval,
            "max_recent_gmm_eval": max_recent_eval,
            "f_of_z_train": f_of_z_train,
            "f_of_z_dev": f_of_z_dev,
        }
