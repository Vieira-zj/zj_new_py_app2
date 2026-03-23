import torch


def get_device() -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU (Metal)
    else:
        device = torch.device("cpu")  # CPU fallback
    return str(device)


def calc_model_memory_size(
    model: torch.nn.Module, input_dtype: torch.dtype = torch.float32
) -> float:
    total_params = 0
    total_grads = 0

    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size

    total_buffers = sum(buf.numel() for buf in model.buffers())

    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    total_memory_gb = total_memory_bytes / (1024**3)
    return total_memory_gb


def main():
    print("pytorch version:", torch.__version__)
    print("using device:", get_device())


if __name__ == "__main__":
    main()
