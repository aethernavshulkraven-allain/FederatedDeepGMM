# import torch
# from torch import nn

# from ...core.alg_frame.client_trainer import ClientTrainer
# from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
# import logging
# import copy
# import logging
# import random
# import math
# from optimizers import fedoptimizer
# import itertools
# # from functorch import grad_and_value, make_functional, vmap


# class ModelTrainerCLS(ClientTrainer):
#     def get_g_model_params(self):
#         return self.g.state_dict()
    
#     def get_f_model_params(self):
#         return self.f.state_dict()
    
#     def get_model_params(self):
#         return self.reg_model.cpu().state_dict()

#     def set_model_params(self, model_parameters):
#         self.reg_model.load_state_dict(model_parameters)
#         self.reg_model = self.reg_model.train()  

#     def set_g_model_params(self, model_parameters):
#         new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
#         self.g.load_state_dict(new_state_dict)
#         self.g = self.g.train()
        
#     def set_f_model_params(self, model_parameters):
#         new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
#         self.f.load_state_dict(new_state_dict)
#         self.f = self.f.train()
        
#     def train(self, client_data, device, args):
#         model = self.reg_model
#         # model = model.load_state_dict(self.get_model_params())
#         model.to(device)
#         model.train()

#         # train and update
#         criterion = nn.MSELoss().to(device)  # pylint: disable=E1102
#         if args.client_optimizer == "sgd":
#             optimizer = torch.optim.SGD(
#                 filter(lambda p: p.requires_grad, self.reg_model.parameters()),
#                 lr=args.learning_rate,
#             )
#         else:
#             optimizer = torch.optim.Adam(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#                 weight_decay=args.weight_decay,
#                 amsgrad=True,
#             )

#         epoch_loss = []
#         for epoch in range(args.epochs):
#             batch_loss = []

#             for epoch in range(args.epochs):
#                 for batch in client_data:
#                     x_batch = batch[2]
#                     y_batch = batch[3]
#                     model.zero_grad()
#                     preds = torch.squeeze(model(x_batch))
#                     truee = torch.squeeze(y_batch)
#                     loss = criterion(preds, truee)  # pylint: disable=E1102
#                     loss.backward()
#                     optimizer.step()

#                 # Uncommet this following line to avoid nan loss
#                 # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 # logging.info(
#                 #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 #         epoch,
#                 #         (batch_idx + 1) * args.batch_size,
#                 #         len(train_data) * args.batch_size,
#                 #         100.0 * (batch_idx + 1) / len(train_data),
#                 #         loss.item(),
#                 #     )
#                 # )

#                 batch_loss.append(loss.item())
#             if len(batch_loss) == 0:
#                 epoch_loss.append(0.0)
#             else:
#                 epoch_loss.append(sum(batch_loss) / len(batch_loss))
#             # logging.info(
#             #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
#             #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
#             #     )
#             # )
    

#     # Function to get a snapshot of the model parameters
#     def get_params_snapshot(self, model):
#         return {name: param.clone() for name, param in model.named_parameters()}

#     def compare_params(self, initial_params, model, model_name):
#         changed = False
#         for name, initial_param in initial_params.items():
#             current_param = model.state_dict()[name]
#             if not torch.equal(current_param, initial_param):
#                 changed = True
#                 print(f"Parameter {name} of model {model_name} has changed.")
#         if not changed:
#             print(f"No parameters of model {model_name} have changed.")
    
        
#     def train_gmm(self, client_data, device, args):
#         g = self.g
#         f = self.f
        
#         g.to(device)
#         f.to(device)
#         g.train()
#         f.train()
        
#     # Snapshot of parameters before training
#         # initial_g_params = self.get_params_snapshot(g)
#         # initial_f_params = self.get_params_snapshot(f)
        
#     # loop through training data
#         for epoch in range(args.epochs):
#             for batch in client_data:
#                 x_batch = batch[2]
#                 y_batch = batch[3]
#                 z_batch = batch[4]
#                 g_obj, f_obj = self.game_objective.calc_objective(
#                    g,f, x_batch, z_batch, y_batch)
#                 # do single step optimization on f and g
#                 # final_g.zero_grad()
#                 self.g_optimizer.zero_grad()
#                 # optimizer_g.zero_grad()
#                 g_obj.backward(retain_graph=True)
#                 # final_g.step()
#                 self.g_optimizer.step()
#                 # optimizer_g.step()

#                 self.f_optimizer.zero_grad()
#                 # optimizer_f.zero_grad()
#                 # final_f.zero_grad()
#                 f_obj.backward()
#                 # final_f.step()
#                 self.f_optimizer.step()
#                 # optimizer_f.step()
#             # scheduler_g.step()
#             # scheduler_f.step()
                
#         # self.compare_params(initial_g_params, g, "g")
#         # self.compare_params(initial_f_params, f, "f")
        
#         self.set_g_model_params(g.state_dict())
#         self.set_f_model_params(f.state_dict())
        
#     def train_iterations(self, train_data, device, args):
#         model = self.model

#         model.to(device)
#         model.train()

#         # train and update
#         criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
#         if args.client_optimizer == "sgd":
#             optimizer = torch.optim.SGD(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#             )
#         else:
#             optimizer = torch.optim.Adam(
#                 filter(lambda p: p.requires_grad, self.model.parameters()),
#                 lr=args.learning_rate,
#                 weight_decay=args.weight_decay,
#                 amsgrad=True,
#             )

#         epoch_loss = []

#         current_steps = 0
#         current_epoch = 0
#         while current_steps < args.local_iterations:
#             batch_loss = []
#             for batch_idx, (x, labels) in enumerate(train_data):
#                 x, labels = x.to(device), labels.to(device)
#                 model.zero_grad()
#                 log_probs = model(x)
#                 labels = labels.long()
#                 loss = criterion(log_probs, labels)  # pylint: disable=E1102
#                 loss.backward()

#                 # Uncommet this following line to avoid nan loss
#                 # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 optimizer.step()
#                 # logging.info(
#                 #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 #         epoch,
#                 #         (batch_idx + 1) * args.batch_size,
#                 #         len(train_data) * args.batch_size,
#                 #         100.0 * (batch_idx + 1) / len(train_data),
#                 #         loss.item(),
#                 #     )
#                 # )
#                 batch_loss.append(loss.item())
#                 current_steps += 1
#                 if current_steps == args.local_iterations:
#                     break
#             current_epoch += 1
#             epoch_loss.append(sum(batch_loss) / len(batch_loss))
#             logging.info(
#                 "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
#                     self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
#                 )
#             )

#     def test(self, test_data, device, args):
#         model = self.model

#         model.to(device)
#         model.eval()

#         metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

#         criterion = nn.CrossEntropyLoss().to(device)

#         with torch.no_grad():
#             for batch_idx, (x, target) in enumerate(test_data):
#                 x = x.to(device)
#                 target = target.to(device)
#                 pred = model(x)
#                 target = target.long()
#                 loss = criterion(pred, target)  # pylint: disable=E1102

#                 _, predicted = torch.max(pred, -1)
#                 correct = predicted.eq(target).sum()

#                 metrics["test_correct"] += correct.item()
#                 metrics["test_loss"] += loss.item() * target.size(0)
#                 metrics["test_total"] += target.size(0)
#         return metrics
import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging
import random
import math
from optimizers import fedoptimizer
import itertools
# from functorch import grad_and_value, make_functional, vmap


class ModelTrainerCLS(ClientTrainer):
    def get_g_model_params(self):
        return self.g.state_dict()
    
    def get_f_model_params(self):
        return self.f.state_dict()
    
    def get_model_params(self):
        return self.reg_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.reg_model.load_state_dict(model_parameters)
        self.reg_model = self.reg_model.train()  

    def set_g_model_params(self, model_parameters):
        new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
        self.g.load_state_dict(new_state_dict)
        self.g = self.g.train()
        
    def set_f_model_params(self, model_parameters):
        new_state_dict = {k.replace('_module.', ''): v for k, v in model_parameters.items()}
        self.f.load_state_dict(new_state_dict)
        self.f = self.f.train()
        
    def train(self, client_data, device, args):
        model = self.reg_model
        # model = model.load_state_dict(self.get_model_params())
        model.to(device)
        model.train()

        # train and update
        criterion = nn.MSELoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for epoch in range(args.epochs):
                for batch in client_data:
                    x_batch = batch[2]
                    y_batch = batch[3]
                    model.zero_grad()
                    preds = torch.squeeze(model(x_batch))
                    truee = torch.squeeze(y_batch)
                    loss = criterion(preds, truee)  # pylint: disable=E1102
                    loss.backward()
                    optimizer.step()

                # Uncommet this following line to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )
    

    # Function to get a snapshot of the model parameters
    def get_params_snapshot(self, model):
        return {name: param.clone() for name, param in model.named_parameters()}

    def compare_params(self, initial_params, model, model_name):
        changed = False
        for name, initial_param in initial_params.items():
            current_param = model.state_dict()[name]
            if not torch.equal(current_param, initial_param):
                changed = True
                print(f"Parameter {name} of model {model_name} has changed.")
        if not changed:
            print(f"No parameters of model {model_name} have changed.")
    
        
    def train_gmm(self, client_data, device, args):
        g = self.g
        f = self.f
        
        # Clear optimizer states because the model weights were just updated with global weights!
        # This is critical for stateful optimizers like OGDA, otherwise prev_grad is stale.
        self.g_optimizer.state.clear()
        self.f_optimizer.state.clear()
        g.to(device)
        f.to(device)
        g.train()
        f.train()
        
        
    # Snapshot of parameters before training
        # initial_g_params = self.get_params_snapshot(g)
        # initial_f_params = self.get_params_snapshot(f)
        
    # loop through training data
        for epoch in range(args.epochs):
            for batch in client_data:
                x_batch = batch[2]
                y_batch = batch[3]
                z_batch = batch[4]
                g_obj, f_obj = self.game_objective.calc_objective(
                   g,f, x_batch, z_batch, y_batch)
                # do single step optimization on f and g
                # final_g.zero_grad()
                self.g_optimizer.zero_grad()
                # optimizer_g.zero_grad()
                g_obj.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
                # final_g.step()
                self.g_optimizer.step()
                # optimizer_g.step()

                self.f_optimizer.zero_grad()
                # optimizer_f.zero_grad()
                # final_f.zero_grad()
                f_obj.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
                # final_f.step()
                self.f_optimizer.step()
                # optimizer_f.step()
            # scheduler_g.step()
            # scheduler_f.step()
                
        # self.compare_params(initial_g_params, g, "g")
        # self.compare_params(initial_f_params, f, "f")
        
        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())

    @staticmethod
    def _rademacher_directions(parameters):
        """Generate independent SPSA directions with entries in {-1, +1}."""
        return [
            torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            for param in parameters
        ]

    @staticmethod
    def _apply_perturbation(parameters, directions, scale):
        with torch.no_grad():
            for param, direction in zip(parameters, directions):
                param.add_(direction, alpha=scale)

    @staticmethod
    def _clip_and_apply_zo_update(parameters, estimates, learning_rate, max_norm=1.0):
        """Apply an SGD step using estimated gradients without backward()."""
        if not estimates:
            return
        total_norm_sq = sum(estimate.pow(2).sum() for estimate in estimates)
        total_norm = total_norm_sq.sqrt()
        clip_scale = min(1.0, max_norm / (total_norm.item() + 1e-12))
        with torch.no_grad():
            for param, estimate in zip(parameters, estimates):
                param.add_(estimate, alpha=-learning_rate * clip_scale)

    def train_gmm_zo(self, client_data, device, args):
        """Forward-only SPSA updates for phase two of server FedZO-EG.

        Both player blocks are perturbed simultaneously. Independent Rademacher
        directions isolate each block in expectation and need two objective
        forward evaluations per direction.
        """
        g = self.g.to(device)
        f = self.f.to(device)
        g.train()
        f.train()

        mu = float(getattr(args, "zo_mu", 1e-3))
        num_directions = int(getattr(args, "zo_num_directions", 1))
        if mu <= 0.0:
            raise ValueError("zo_mu must be positive")
        if num_directions < 1:
            raise ValueError("zo_num_directions must be at least 1")

        g_lr = float(self.g_optimizer.param_groups[0]["lr"])
        f_lr = float(self.f_optimizer.param_groups[0]["lr"])
        g_parameters = [p for p in g.parameters() if p.requires_grad]
        f_parameters = [p for p in f.parameters() if p.requires_grad]

        for _ in range(args.epochs):
            for batch in client_data:
                x_batch, y_batch, z_batch = batch[2], batch[3], batch[4]
                g_estimates = [torch.zeros_like(p) for p in g_parameters]
                f_estimates = [torch.zeros_like(p) for p in f_parameters]

                for _ in range(num_directions):
                    g_directions = self._rademacher_directions(g_parameters)
                    f_directions = self._rademacher_directions(f_parameters)

                    self._apply_perturbation(g_parameters, g_directions, mu)
                    self._apply_perturbation(f_parameters, f_directions, mu)
                    with torch.no_grad():
                        g_plus, f_plus = self.game_objective.calc_objective(
                            g, f, x_batch, z_batch, y_batch)

                    self._apply_perturbation(g_parameters, g_directions, -2.0 * mu)
                    self._apply_perturbation(f_parameters, f_directions, -2.0 * mu)
                    with torch.no_grad():
                        g_minus, f_minus = self.game_objective.calc_objective(
                            g, f, x_batch, z_batch, y_batch)

                    # Restore the unperturbed look-ahead parameters.
                    self._apply_perturbation(g_parameters, g_directions, mu)
                    self._apply_perturbation(f_parameters, f_directions, mu)

                    g_coeff = (g_plus.item() - g_minus.item()) / (2.0 * mu)
                    f_coeff = (f_plus.item() - f_minus.item()) / (2.0 * mu)
                    for estimate, direction in zip(g_estimates, g_directions):
                        estimate.add_(direction, alpha=g_coeff / num_directions)
                    for estimate, direction in zip(f_estimates, f_directions):
                        estimate.add_(direction, alpha=f_coeff / num_directions)

                self._clip_and_apply_zo_update(g_parameters, g_estimates, g_lr)
                self._clip_and_apply_zo_update(f_parameters, f_estimates, f_lr)

        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())

    def train_iterations(self, train_data, device, args):
        model = self.reg_model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.reg_model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
