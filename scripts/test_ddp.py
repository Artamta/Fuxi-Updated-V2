import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] on GPU {local_rank}", flush=True)

    x = torch.tensor([rank], device="cuda")
    dist.all_reduce(x)

    print(f"[Rank {rank}] total = {x.item()}", flush=True)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()