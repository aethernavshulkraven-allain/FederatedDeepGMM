# `fedavg_api.py` Call Tree

This is a static call/dependency tree for
`fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py`.
It follows executable code, not commented-out experiments. Conditional paths are
labelled; the current YAML selects the default classification trainer and
`client_optimizer: fed_eg`.

```text
fedml/simulation/simulator.py
└── SimulatorSingleProcess.__init__()
    └── fedml/simulation/sp/fedavg/fedavg_api.py
        └── FedAvgAPI.__init__()
            ├── game_objectives/simple_moment_objective.py
            │   └── OptimalMomentObjective()
            ├── optimizers/optimizer_factory.py
            │   └── OptimizerFactory(...)
            │       ├── optimizers/ogda.py -> OGDA                 [client_optimizer=ogda]
            │       └── optimizers/Customsgd.py -> CustomSGD       [all other values]
            ├── model_selection/simple_model_eval.py
            │   └── SGDSimpleModelEval()
            ├── model_selection/learning_eval_nostop.py
            │   └── FHistoryLearningEvalSGDNoStop()
            │       └── OptimalMomentObjective.calc_objective()
            ├── model_selection_class.py
            │   └── FHistoryModelSelectionV3.do_model_selection()
            │       ├── OptimizerFactory.__call__()
            │       ├── FHistoryLearningEvalSGDNoStop.eval()
            │       └── game_objectives/approximate_psi_objective.py
            │           └── max_approx_psi_eval()
            ├── fedml/ml/trainer/trainer_creator.py
            │   └── create_model_trainer()
            │       └── fedml/ml/trainer/my_model_trainer_classification.py
            │           └── ModelTrainerCLS()                      [current/default datasets]
            ├── fedml/simulation/sp/fedavg/client.py
            │   └── Client() for each serial client slot            [SP mode]
            ├── fedml/simulation/sp/fedavg/multiprocess_client.py
            │   ├── MultiprocessClientExecutor()                    [multi-GPU mode]
            │   │   └── one persistent spawned worker per GPU
            │   └── SingleGPUMultiprocessClientExecutor()           [one-GPU process mode]
            │       └── persistent spawned workers sharing one GPU
            └── fedml/simulation/sp/fedavg/single_gpu_client.py
                └── SingleGPUClientExecutor()                      [single-GPU mode]
                    └── one isolated g/f trainer and CUDA stream per concurrent slot

fedml/simulation/simulator.py
└── SimulatorSingleProcess.run()
    └── FedAvgAPI.train()
        ├── FedAvgAPI._client_sampling()
        ├── FedAvgAPI._run_primary_client_updates()
        │   ├── SP: Client.update_local_dataset()/train()
        │   ├── multi-GPU: MultiprocessClientExecutor.run()
        │   │   └── spawned worker -> Client.train()
        │   ├── one-GPU processes: SingleGPUMultiprocessClientExecutor.run()
        │   │   └── same-GPU spawned worker -> Client.train()
        │   └── single-GPU: SingleGPUClientExecutor.run()
        │       └── stream slot -> Client.train()
        │           └── ModelTrainerCLS.train_gmm()
        │               ├── OptimalMomentObjective.calc_objective()
        │               └── CustomSGD.step() or OGDA.step()
        ├── FedAvgAPI._aggregate()
        │   └── experiment_utils.weighted_average_state_dicts()
        │       ├── weighted aggregation for floating tensors
        │       └── deterministic maximum for nonfloating counters
        ├── experiment_utils.apply_parameter_server_update()
        │   ├── server interpolation/optimism on named parameters only
        │   └── direct aggregated state for model buffers
        ├── synchronized second client phase                    [fed_eg or fed_zo_eg]
        │   ├── serial/worker/stream -> Client.train()                                  [fed_eg]
        │   │   └── ModelTrainerCLS.train_gmm()
        │   └── serial/worker/stream -> Client.train_zo()                               [fed_zo_eg]
        │       └── ModelTrainerCLS.train_gmm_zo()
        ├── experiment_utils.batchnorm_running_var_min()
        │   └── reject nonfinite or negative BatchNorm running variance
        ├── FedAvgAPI.eval_global_model()
        │   ├── FedAvgAPI.calc_f_g_obj()
        │   │   └── OptimalMomentObjective.calc_objective()
        │   └── game_objectives/approximate_psi_objective.py
        │       └── approx_psi_eval()
        ├── log_results_to_csv() -> csv/<optimizer>_<dataset>newtrial.csv
        ├── torch.save() -> checkpoints/<optimizer>_<dataset>_round_<n>.pt
        └── plotting.py -> PlotElement                         [video plotting enabled only]
```

## Trainer Selection Branches

`trainer_creator.py` selects `ModelTrainerCLS` for the repository's configured
datasets (`abs`, `step`, `linear`, MNIST/FEMNIST, and CIFAR variants). Its other
branches call `my_model_trainer_tag_prediction.py` for `stackoverflow_lr` and
`my_model_trainer_nwp.py` for `fed_shakespeare` or `stackoverflow_nwp`; these are
not active under the current configuration.

## Direct Imports Not on the Active Call Path

`fedavg_api.py` imports `OAdam`, `GradientDecentSimpleModelEval`, `Adam`, and
`torch.optim.sgd`, but the executable statements currently do not call them.
They are therefore excluded from the runtime tree above.
