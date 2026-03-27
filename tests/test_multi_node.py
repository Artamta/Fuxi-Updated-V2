import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print(f"Rank {rank}/{world_size} | Local rank {local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)

    x = torch.tensor([rank], device="cuda")
    dist.all_reduce(x)

    print(f"[Rank {rank}] result: {x.item()}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()