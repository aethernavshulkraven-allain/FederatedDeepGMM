"""Persistent multi-GPU workers for independent federated client updates."""

import atexit
import copy
import logging
import queue
import traceback

import torch
import torch.multiprocessing as multiprocessing

from .client import Client


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_cpu(item) for item in value]
        return type(value)(converted)
    return value



def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_device(item, device) for item in value]
        return type(value)(converted)
    return value

def _move_trainer_to_cpu(trainer):
    trainer.g.cpu()
    trainer.f.cpu()
    trainer.reg_model.cpu()
    trainer.g_optimizer.state.clear()
    trainer.f_optimizer.state.clear()
    return trainer


def _worker_loop(worker_id, gpu_id, trainer, args, task_queue, result_queue):
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        trainer = _move_trainer_to_cpu(trainer)

        while True:
            task = task_queue.get()
            if task is None:
                return

            task_id = task["task_id"]
            try:
                client = Client(
                    task["client_idx"],
                    _to_device(task["train_data"], device),
                    None,
                    task["sample_number"],
                    args,
                    device,
                    trainer,
                )
                phase = task["phase"]
                if phase == "primary":
                    gmm_weights = client.train(task["g_global"], task["f_global"])
                    reg_weights = client.train_reg(task["reg_global"])
                    payload = {
                        "gmm": _to_cpu(gmm_weights),
                        "reg": _to_cpu(reg_weights),
                    }
                elif phase == "correction":
                    if task["use_zeroth_order"]:
                        weights = client.train_zo(task["g_global"], task["f_global"])
                    else:
                        weights = client.train(task["g_global"], task["f_global"])
                    payload = {"gmm": _to_cpu(weights)}
                else:
                    raise ValueError(f"Unknown client phase: {phase}")

                result_queue.put(
                    {
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "payload": payload,
                        "error": None,
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "payload": None,
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        result_queue.put(
            {
                "task_id": None,
                "worker_id": worker_id,
                "payload": None,
                "error": traceback.format_exc(),
            }
        )


class MultiprocessClientExecutor:
    """Run independent client updates concurrently and return ordered results."""

    def __init__(self, trainer, args, gpu_ids):
        self.context = multiprocessing.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.task_queues = []
        self.processes = []
        self.next_task_id = 0
        self._closed = False
        trainer_template = _move_trainer_to_cpu(copy.deepcopy(trainer))

        for worker_id, gpu_id in enumerate(gpu_ids):
            task_queue = self.context.Queue(maxsize=1)
            process = self.context.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    gpu_id,
                    copy.deepcopy(trainer_template),
                    copy.deepcopy(args),
                    task_queue,
                    self.result_queue,
                ),
                daemon=True,
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)

        self.worker_count = len(self.processes)
        atexit.register(self.close)
        logging.info(
            "Started %d federated client workers on logical GPUs %s",
            self.worker_count,
            gpu_ids,
        )

    def run(self, tasks):
        if not tasks:
            return []

        pending = list(enumerate(tasks))
        active = {}
        results = [None] * len(tasks)

        for worker_id in range(min(self.worker_count, len(pending))):
            result_index, task = pending.pop(0)
            self._submit(worker_id, result_index, task, active)

        completed = 0
        while completed < len(tasks):
            try:
                result = self.result_queue.get(timeout=30)
            except queue.Empty:
                dead = [
                    index
                    for index, process in enumerate(self.processes)
                    if not process.is_alive()
                ]
                if dead:
                    raise RuntimeError(f"Federated client workers exited unexpectedly: {dead}")
                continue

            if result["task_id"] is None:
                raise RuntimeError(
                    f"Client worker {result['worker_id']} failed during "
                    f"startup:\n{result['error']}"
                )
            result_index, worker_id = active.pop(result["task_id"])
            if result["error"] is not None:
                raise RuntimeError(
                    f"Client worker {worker_id} failed:\n{result['error']}"
                )
            results[result_index] = result["payload"]
            completed += 1

            if pending:
                next_index, next_task = pending.pop(0)
                self._submit(worker_id, next_index, next_task, active)

        return results

    def _submit(self, worker_id, result_index, task, active):
        task = dict(task)
        task_id = self.next_task_id
        self.next_task_id += 1
        task["task_id"] = task_id
        active[task_id] = (result_index, worker_id)
        self.task_queues[worker_id].put(task)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for task_queue in self.task_queues:
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for task_queue in self.task_queues:
            task_queue.close()
        self.result_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
