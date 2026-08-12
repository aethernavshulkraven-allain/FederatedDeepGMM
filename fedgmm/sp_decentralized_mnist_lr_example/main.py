import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ["FEDML_USING_MLOPS"] = "false"

import fedml
import torch
from fedml import FedMLRunner

current_file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_dir)
if __name__ == "__main__":
    # init FedML framework
    print("DEBUG: Calling fedml.init()...")
    args = fedml.init()
    print("DEBUG: fedml.init() successful.")

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)
    
    model = fedml.model.create(args, output_dim)
    if args.federated_optimizer == "FedAvg":
        # The supported DeepGMM FedAvg path uses only structural g and critic f.
        # Keep the legacy third model available to other vendored coordinators.
        model = model[:2]

    # # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    
    fedml_runner.run()