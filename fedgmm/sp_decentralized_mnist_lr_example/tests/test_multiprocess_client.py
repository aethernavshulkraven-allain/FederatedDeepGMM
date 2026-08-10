from types import SimpleNamespace

import torch

from fedml.simulation.sp.fedavg.multiprocess_client import _to_cpu, _to_device


def test_nested_tensor_device_conversion_preserves_structure():
    value = {
        "weights": [torch.tensor([1.0]), (torch.tensor([2.0]),)],
        "metadata": "client",
    }

    cpu_value = _to_cpu(value)
    converted = _to_device(cpu_value, torch.device("cpu"))

    assert isinstance(converted, dict)
    assert isinstance(converted["weights"], list)
    assert isinstance(converted["weights"][1], tuple)
    assert converted["metadata"] == "client"
    assert converted["weights"][0].device.type == "cpu"
    assert converted["weights"][1][0].item() == 2.0


def test_to_cpu_detaches_worker_results():
    tensor = torch.tensor([3.0], requires_grad=True)

    result = _to_cpu(SimpleNamespace(value=tensor).value)

    assert result.device.type == "cpu"
    assert not result.requires_grad
