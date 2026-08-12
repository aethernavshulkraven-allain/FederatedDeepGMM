import os

# Every model trained through this entry point is a tiny 1-2 hidden-layer MLP
# over at most a few thousand rows; PyTorch's CPU default is one BLAS thread
# per core, which on a shared multi-user machine lets one process grab dozens
# of cores for matmuls too small to benefit (measured: ~29 cores/process,
# almost pure thread-contention overhead, no speedup). Must be set before
# `import fedml`, which transitively imports numpy/torch and initializes
# their thread pools at that point.
_DEFAULT_CPU_THREADS = "4"
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_var, _DEFAULT_CPU_THREADS)

import fedml
from fedml import FedMLRunner
import torch
from experiment_utils import RuntimeProfiler, get_effective_config, run_dir_from_config

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
# Only claim all local GPUs when nothing upstream has scoped us to a subset.
# Under the gpu-broker (`gpurun`) CUDA_VISIBLE_DEVICES is preset to the GPUs
# actually allocated to this job; overwriting it would let the job run on a GPU
# belonging to another user, which the server policy forbids.
if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['FEDML_USING_MLOPS'] = 'false'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_dir)
if __name__ == "__main__":
    # init FedML framework
    print("DEBUG: Calling fedml.init()...")
    args = fedml.init()
    print("DEBUG: fedml.init() successful.")
    profile_config = get_effective_config(args)
    runtime_profiler = RuntimeProfiler.from_config(profile_config)
    runtime_profiler.configure(profile_config, run_dir_from_config(profile_config))
    setattr(args, "_fedgmm_runtime_profiler", runtime_profiler)
    runtime_profiler.record_environment(profile_config)
    runtime_profiler.start_telemetry()

    try:
        # init device
        with runtime_profiler.span("device_init"):
            device = fedml.device.get_device(args)
        runtime_profiler.record_environment(profile_config, device=device)

        # load data
        with runtime_profiler.span("data_load"):
            dataset, output_dim = fedml.data.load(args)

        with runtime_profiler.span("model_create"):
            model = fedml.model.create(args, output_dim)
            if not bool(getattr(args, "auxiliary_regression", True)):
                # auxiliary_regression=false: model_hub already returned an
                # empty reg_models list in this case, so this just drops the
                # placeholder. FedAvgAPI itself re-derives the same trim from
                # this same flag (see its skip_auxiliary_regression guard).
                model = model[:2]

        # # start training
        with runtime_profiler.span("runner_init"):
            fedml_runner = FedMLRunner(args, device, dataset, model)

        with runtime_profiler.span("runner_run"):
            fedml_runner.run()
        runtime_profiler.stop(extra={"exit_status": "completed"})
    except Exception as exc:
        runtime_profiler.stop(extra={"exit_status": "exception", "exception": repr(exc)})
        raise
