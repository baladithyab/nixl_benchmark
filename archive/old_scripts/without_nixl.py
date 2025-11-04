import io
import time
import uuid

import torch


def create_inference_state():
    state = {}

    # 512 is an average number, this size changes as the user interact with the model
    num_keys = 512
    for _ in range(num_keys):
        key = str(uuid.uuid4())
        tensor1 = torch.rand(1, 3, 512, 512).half().cuda()
        tensor2 = torch.rand(3, 256, 32, 32).half().cuda()
        tensor2_slices = {i: tensor2[i] for i in range(tensor2.size(0))}
        value = {
            "tensor1": tensor1,
            "tensor2": tensor2,
            "tensor2_slices": tensor2_slices,
        }
        state[key] = value

    return state


def gpu_to_cpu(state):
    output = io.BytesIO()
    torch.save(state, output)
    return output.getvalue()


def cpu_to_gpu(serialized_state):
    return torch.load(io.BytesIO(serialized_state))


if __name__ == "__main__":
    state = create_inference_state()

    time_b = time.time()
    serialized_state = gpu_to_cpu(state)
    time_e = time.time()
    print(f"time to gpu_to_cpu: {time_e - time_b:.2f} sec")

    time_b = time.time()
    state_loaded = cpu_to_gpu(serialized_state)
    time_e = time.time()
    print(f"time to cpu_to_gpu: {time_e - time_b:.2f} sec")
