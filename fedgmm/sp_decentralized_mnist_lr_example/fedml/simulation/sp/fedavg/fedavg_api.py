import copy
import logging
import math
import csv
import os
import time
import numpy 
import torch
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from .multiprocess_client import MultiprocessClientExecutor, _to_cpu, _to_device
# from fedml.core.dp.mechanisms import 
# from opacus.layers import 
from model_selection.simple_model_eval import GradientDecentSimpleModelEval
from model_selection_class import FHistoryModelSelectionV3
from game_objectives.simple_moment_objective import OptimalMomentObjective
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
        }
        for arg_name, default_value in research_args.items():
            if not hasattr(self.args, arg_name):
                setattr(self.args, arg_name, default_value)
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
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.val_data_num_in_total = val_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict


        logging.info("model = {}".format(model))
        self.model = model
        self.device = device

        # Move models to device
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
        # g_learning_rates = [0.010, 0.050, 0.020]
        ##g_learning_rates =[0.01, 0.001,0.0001,0.0005]
        # g_learning_rates = [0.0005]
        # g_learning_rates=[0.1,0.2,0.5]
        # g_learning_rates =[0.00010, 0.000050, 0.000020]
        # Use the learning rate from the YAML configuration
        g_learning_rates = [self.args.learning_rate]
        game_objectives = [ 
            OptimalMomentObjective(),
        ]
        learning_setups = []
        # Dynamically adjust Critic multiplier based on dataset complexity
        # CNN critics (in z and xz scenarios) are very powerful and need a lower multiplier to avoid NaN
        critic_multiplier = 20.0
        if args.dataset in ['linear', 'abs', 'sin', 'step', 'zoo']:
            critic_multiplier = 10.0
        if args.dataset in ['mnist_z', 'femnist_z', 'mnist_xz', 'femnist_xz','cifar10_z', 'cifar10_xz', 'cifar_xz']:
            critic_multiplier = 5.0  # CNN Critic is powerful enough without a high LR
        elif args.dataset in ['mnist_x', 'femnist_x','cifar10_x', 'cifar_x']:
            critic_multiplier = 3.0  # Balanced approach
       
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
                            # OGDA, lr=20.0*float(g_lr)),
                            OGDA, lr=critic_multiplier*float(g_lr)), # Adjusted
                        "game_objective": game_objective
                    }
                else:
                    learning_setup = {
                        "g_optimizer_factory": OptimizerFactory(
                            CustomSGD, lr=float(g_lr), momentum=0.0),
                        "f_optimizer_factory": OptimizerFactory(
                            # CustomSGD, lr=20.0*float(g_lr), momentum=0.0),
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
            max_num_epoch=100, max_no_progress=10, batch_size=200, eval_freq=1)
        f_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=100, max_no_progress=10, batch_size=200, eval_freq=1)
        learning_eval = FHistoryLearningEvalSGDNoStop(
            num_epochs=60, eval_freq=1, batch_size=200)
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=model[0],
            f_model_list=model[1],
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_max_no_progress=10, psi_eval_burn_in=30,
        )
        self.default_g_opt_factory = default_g_opt_factory
        # g_simple_model_eval = SGDSimpleModelEval()
        # f_simple_model_eval = SGDSimpleModelEval()
        # learning_eval = FHistoryLearningEvalSGDNoStop(num_epochs=args.epochs_model_selection, eval_freq=args.eval_freq, print_freq=args.print_freq, batch_size=args.batch_size)
        self.reg_model = model[2][0]
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
        g_global, f_global, learning_args, dev_f_collection, e_dev_tilde = \
            self.model_selection.do_model_selection(
                x_train=train_data_global.x, z_train=train_data_global.z, y_train=train_data_global.y,
                x_dev=val_data_global.x, z_dev=val_data_global.z, y_dev=val_data_global.y, verbose=True)
        
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
        
        self.model_trainer = create_model_trainer([g_global, f_global, model[2][0]], learning_args, args)

        self.client_executor = self._create_client_executor()
        if self.client_executor is None:
            self._setup_clients(
                train_data_local_num_dict, train_data_local_dict,
                test_data_local_dict, self.model_trainer,
            )

    def _create_client_executor(self):
        if not self.args.enable_multiprocessing:
            return None
        if not torch.cuda.is_available():
            logging.warning("Multiprocessing requested without CUDA; using SP execution")
            return None

        gpu_ids = self.args.multiprocessing_gpu_ids
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, str):
            gpu_ids = [int(value.strip()) for value in gpu_ids.split(",") if value.strip()]
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
            return None
        return MultiprocessClientExecutor(self.model_trainer, self.args, gpu_ids)

    @staticmethod
    def _materialize_client_data(data_loader):
        return [_to_cpu(batch) for batch in data_loader]

    def _run_primary_client_updates(self, client_indexes, g_global, f_global, reg_global):
        phase_start = time.perf_counter()
        if self.client_executor is None:
            w_locals = []
            w_locals_reg = []
            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )
                weights = client.train(copy.deepcopy(g_global), copy.deepcopy(f_global))
                reg_weights = client.train_reg(copy.deepcopy(reg_global))
                sample_number = client.get_sample_number()
                w_locals.append((sample_number, copy.deepcopy(weights)))
                w_locals_reg.append((sample_number, copy.deepcopy(reg_weights)))
            logging.info(
                "Client primary phase mode=sp clients=%d elapsed_seconds=%.6f",
                len(client_indexes), time.perf_counter() - phase_start,
            )
            return w_locals, w_locals_reg

        tasks = []
        cpu_g_global = _to_cpu(g_global)
        cpu_f_global = _to_cpu(f_global)
        cpu_reg_global = _to_cpu(reg_global)
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
                    "reg_global": cpu_reg_global,
                }
            )
        results = self.client_executor.run(tasks)
        w_locals = []
        w_locals_reg = []
        for client_idx, result in zip(client_indexes, results):
            sample_number = self.train_data_local_num_dict[client_idx]
            w_locals.append(
                (sample_number, _to_device(result["gmm"], self.device))
            )
            w_locals_reg.append((sample_number, result["reg"]))
        logging.info(
            "Client primary phase mode=mp clients=%d workers=%d elapsed_seconds=%.6f",
            len(client_indexes), self.client_executor.worker_count,
            time.perf_counter() - phase_start,
        )
        return w_locals, w_locals_reg

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
            "Client correction phase mode=mp clients=%d workers=%d elapsed_seconds=%.6f",
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
        reg_global = self.model_trainer.get_model_params() 
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

        for round_idx in range(start_round, self.args.comm_round):

            # logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            w_locals_reg = []
            w_locals_prev = []
            # obj_sum=[]
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            # logging.info("client_indexes = " + str(client_indexes))

            w_locals, w_locals_reg = self._run_primary_client_updates(
                client_indexes, g_global, f_global, reg_global
            )
            # update global weights
            w_agg = self._aggregate(w_locals)
            
            if self.args.client_optimizer in ("fed_eg", "fed_zo_eg"):
                # Phase 1 has evaluated the local SGD/FedAvg map at z_t.
                # Build the server predictor from the aggregated displacement.
                g_base = self.model_trainer.get_g_model_params()
                f_base = self.model_trainer.get_f_model_params()
                predictor_lr = self.args.eg_predictor_server_lr
                corrector_lr = self.args.eg_corrector_server_lr
                if predictor_lr is None:
                    predictor_lr = self.args.server_learning_rate
                if corrector_lr is None:
                    corrector_lr = self.args.server_learning_rate

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

                # Phase 2 uses the same sampled clients and local datasets, now
                # initialized at the globally aggregated look-ahead model.
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
                # Suitable beta (Server LR). 1.0 is standard for FedAvg. 
                # Can be lowered (e.g., 0.5) for more stability.
                server_lr = self.args.server_learning_rate

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
                server_lr = self.args.server_learning_rate

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

            w_global_reg = self._aggregate_reg(w_locals_reg)
            self.model_trainer.set_g_model_params(w_global[0])
            self.model_trainer.set_f_model_params(w_global[1])
            self.model_trainer.set_model_params(w_global_reg)
            # mlops.event("agg", event_started=False, event_value=str(round_idx))

            # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev = self.eval_global_model()
            log_results_to_csv(f"csv/{self.args.client_optimizer}_{self.args.dataset}newtrial.csv", round_idx, mse)
            
            # Save checkpoint every 200 rounds
            if round_idx % 200 == 0:
                checkpoint_path = f"checkpoints/{self.args.client_optimizer}_{self.args.dataset}_round_{round_idx}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'round': round_idx,
                    'g_state_dict': self.g.state_dict(),
                    'f_state_dict': self.f.state_dict(),
                    'mse': mse
                }, checkpoint_path)
            # wandb.log({"round":round_idx,"MSE" :mse})
            # logging.info(f"{round_idx}: {mse:.4f}")
            # print(round_idx,end=" ")
            # print(mse)
            # fedAvg.append(mse)
            if round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    # self._local_test_on_all_clients(round_idx)
                    mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev = self.eval_global_model()
                
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

        # plot relationship between MSE and eval
        # if self.args.video_plotter:
        #     plt.figure()
        #     data = pandas.DataFrame({"eval": self.eval_list, "mse": self.mse_list})
        #     data.plot.scatter(x="eval", y="mse")
        #     plt.savefig("eval_mse.png")
            
        max_i = max(range(len(self.eval_history)), key=lambda i_: self.eval_history[i_])
        if self.args.verbose:
            # print("best iteration:", self.args.eval_freq * max_i)
            pass
            # mlops.log_round_info(self.args.comm_round, round_idx)
        self.model_trainer.set_g_model_params(self.g_state_history[max_i])
        g_final = self.g
        # torch.save(g_final,'/home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/model_step_fedsgd')
        reg_model_final = self.reg_model
        g_final.load_state_dict(self.model_trainer.get_g_model_params())
        reg_model_final.load_state_dict(self.model_trainer.get_model_params())
        g_pred = g_final(self.test_global.x)
        reg_model_final.to(self.device)
        reg_pred = reg_model_final(self.test_global.x)
        # model_linear = torch.load('/home/somya/final/model_oldabs')
        # model_linear_sgd = torch.load('/home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/model_abs')
        # model_linear_sgd_fedavg = torch.load('/home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/model_abs_fedsgd')
        # model_linear_sgd_plain = torch.load('/home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/model_abs_plain')

        # model_linear.to(self.device)
        # model_linear_sgd_fedavg.to(self.device)
        # model_linear_sgd.to(self.device)
        # model_linear_sgd_plain.to(self.device)
        # gmm_pred = model_linear(self.test_global.x)
        # gmm_pred_sgd = model_linear_sgd(self.test_global.x)
        # fedavg_sgd = model_linear_sgd_fedavg(self.test_global.x)
        # sgd_plain = model_linear_sgd_plain(self.test_global.x)
        
        # mse = float(((fedavg_sgd - self.test_global.g) ** 2).mean())
        mse = float(((g_pred - self.test_global.g) ** 2).mean())
        # print("---------------")
        # print("finished running methodology on scenario %s" % self.args.scenario_name)
        print("MSE on test ------------------------------>>>>>>>>>>>>>>>>>>", mse)
        # print("")
        # print("saving results...")
        g_pred = g_pred.detach().cpu().numpy()
        g_true = self.test_global.g.detach().cpu().numpy()
        # gmm_pred = gmm_pred.detach().cpu().numpy()
        reg_pred = reg_pred.detach().cpu().numpy()
        # sgd_plain = sgd_plain.detach().cpu().numpy()
        # gmm_pred_sgd = gmm_pred_sgd.detach().cpu().numpy()
        # fedavg_sgd = fedavg_sgd.detach().cpu().numpy()

        x_sort, g_pred_sort, g_true_sort = prepare_curve_data(
            self.test_global, g_pred, g_true
        )
        x_label =[]
        for i in range(20):
            x_label.append(i)
        
        # Save the data for later plotting
        numpy.save(f"results_{self.args.dataset}_{self.args.client_optimizer}_x.npy", x_sort)
        numpy.save(f"results_{self.args.dataset}_{self.args.client_optimizer}_y_prednewtrial.npy", g_pred_sort)
        numpy.save(f"results_{self.args.dataset}_{self.args.client_optimizer}_y_true.npy", g_true_sort)
        # gmm_true_sort = gmm_pred[indices]
        # gmm_sgd_sort = gmm_pred_sgd[indices]
        # fedavg_sgd_sort = fedavg_sgd[indices]
        # sgd_plain_sort = sgd_plain[indices]
        # for i in range(len(x_sort)):
        #     log_results_to_csv("/home/somya/thesis/new_FedDeepGMM-SGDA.csv", x_sort[i][0], g_pred_sort[i][0])
        #     log_results_to_csv("/home/somya/thesis/new_Actual Causal Effect.csv", x_sort[i][0], g_true_sort[i][0])
        #     log_results_to_csv("/home/somya/thesis/new_DeepGMM-OAdam.csv", x_sort[i][0], gmm_true_sort[i][0])
        #     log_results_to_csv("/home/somya/thesis/new_DeepGMM-SMDA.csv", x_sort[i][0], gmm_sgd_sort[i][0])
        #     log_results_to_csv("/home/somya/thesis/new_FedDeepGMM-SMDA.csv", x_sort[i][0], fedavg_sgd_sort[i][0])
        #     log_results_to_csv("/home/somya/thesis/new_DeepGMM-SGDA.csv", x_sort[i][0], sgd_plain_sort[i][0])
        pred_plot = PlotElement(x_sort, g_pred_sort, "FedDeepGMM-SGDA")
        true_plot = PlotElement(x_sort, g_true_sort, "Actual Causal Effect")
        # gmm_plot = PlotElement(x_sort, gmm_true_sort, "DeepGMM-OAdam")
        # gmm_sgd_plot = PlotElement(x_sort, gmm_sgd_sort, "DeepGMM-SMDA")
        # fedavg_sgd_plot = PlotElement(x_sort,fedavg_sgd_sort,"FedDeepGMM-SMDA")
        # sgd_plain_plot = PlotElement(x_sort,sgd_plain_sort,"DeepGMM-SGDA")

        # plot_Avg = PlotElement(x_label,fedAvg,"FedAvg")
        # reg_NN_plot = PlotElement(x_sort, reg_pred_sort, "Direct predictions from Neural Network")
        
        os.makedirs("plots", exist_ok=True)
        fig, ax = plt.subplots()
        # ax = sgd_plain_plot.plot(ax=ax)
        ax = true_plot.plot(ax=ax, save_path=f'plots/aaaa_{self.args.run_name}_.png')
        # ax = gmm_plot.plot(ax=ax, save_path=f'plots/aaaa_{self.args.run_name}_.png')
        # ax = gmm_sgd_plot.plot(ax=ax, save_path=f'plots/aaaa_{self.args.run_name}_.png')
        # ax = fedavg_sgd_plot.plot(ax=ax, save_path=f'plots/aaaa_{self.args.run_name}_.png')
        ax = pred_plot.plot(ax=ax, save_path=f'plots/aaaa_{self.args.run_name}_.png')
        # ax = reg_NN_plot.plot(ax=ax, save_path=f'plots/aaaacomparison_{self.args.run_name}_.png')
        
        # mlops.log_training_finished_status()
        # mlops.log_aggregation_finished_status()
        
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            numpy.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = numpy.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    def _aggregate_t(self,obj_sum):
        total_sum=0
        for i in obj_sum:
            total_sum+=i
        return total_sum/10
    def _aggregate_reg(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    
    def _aggregate(self, w_locals):
        training_num = sum([num for num, (_) in w_locals])

        (sample_num, (g, f)) = w_locals[0]
        for k in g.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, (local_g, _) = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    g[k] = local_g[k] * w
                else:
                    g[k] += local_g[k] * w
        
        for k in f.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, (_, local_f) = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    f[k] = local_f[k] * w
                else:
                    f[k] += local_f[k] * w
        return [g, f]

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
        g_of_x = None
        f_of_z = None
        obj_total = 0
        for b in range(num_batch):
            if b < num_batch - 1:
                batch_idx = list(range(b*batch_size, (b+1)*batch_size))
            else:
                batch_idx = list(range(b*batch_size, num_data))
            x_batch = x[batch_idx]
            z_batch = z[batch_idx]
            y_batch = y[batch_idx]
            g_obj, _ = self.model_trainer.game_objective.calc_objective(self.model_trainer.g, self.model_trainer.f, x_batch, z_batch, y_batch)
            g_of_x_batch = self.model_trainer.g(x_batch).detach().cpu()
            f_of_z_batch = self.model_trainer.f(z_batch).detach().cpu()
            if b == 0:
                g_of_x = g_of_x_batch
                f_of_z = f_of_z_batch
            else:
                g_of_x = torch.cat([g_of_x, g_of_x_batch], dim=0)
                f_of_z = torch.cat([f_of_z, f_of_z_batch], dim=0)
            obj_total += float(g_obj.detach().cpu()) * len(batch_idx) * 1.0 / num_data
        return g_of_x, f_of_z, float(g_obj.detach().cpu())
    
        
    def eval_global_model(self):
        self.f = self.model_trainer.f.eval()
        self.g = self.model_trainer.g.eval()
        g_of_x_train, f_of_z_train, obj_train = self.calc_f_g_obj(self.train_global)
        g_of_x_dev, f_of_z_dev, obj_dev = self.calc_f_g_obj(self.val_global)
        epsilon_dev = g_of_x_dev - self.val_global.y.cpu()
        epsilon_train = g_of_x_train - self.train_global.y.cpu()
        curr_eval = approx_psi_eval(epsilon_dev, self.dev_f_collection,
                                            self.e_dev_tilde)
        g_error = epsilon_train + self.train_global.y.cpu() - self.train_global.g.cpu()
        mse = float((g_error ** 2).mean())
        self.eval_list.append(curr_eval)
        self.mse_list.append(mse)
        if self.eval_history:
            max_recent_eval = max(self.eval_history)
        else:
            max_recent_eval = float("-inf")
        self.eval_history.append(curr_eval)
        self.epsilon_dev_history.append(epsilon_dev)
        self.epsilon_train_history.append(epsilon_train)
        self.g_state_history.append(copy.deepcopy(self.g.state_dict()))

        self.f = self.f.train()
        self.g = self.g.train()
        self.model_trainer.set_f_model_params(self.f.state_dict())
        self.model_trainer.set_g_model_params(self.g.state_dict())
        return mse, obj_train, obj_dev, curr_eval, max_recent_eval, f_of_z_train, f_of_z_dev