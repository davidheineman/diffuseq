"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

import torch as th
import torch.distributed as dist

import ifcfg

from datetime import timedelta

# Change this to reflect your cluster layout.

def get_ifname():
    return ifcfg.default_interface()["device"]

def setup_dist_old():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
    
    dist.init_process_group(backend=backend, init_method="env://")

def setup_dist():
    """
    Setup distributed torch with SLURM
    """
    # GLOO does not work with CUDA with this code for some incredibly vague reason...
    backend = "nccl"

    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    # master_port = int(os.environ.get("MASTER_PORT", 8738))
    # master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    # Need to override address and port because these arguments get botched
    # by SLURM when the node is created
    master_addr = "127.0.0.1"
    master_port = 8739

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_PORT"] = str(master_port)

    print(f"Overriding local rank {local_rank} with {world_rank}")
    os.environ['LOCAL_RANK'] = str(world_rank) # !! should be local_rank

    os.environ['NCCL_BLOCKING_WAIT'] = str(1)

    print(f"World rank: {world_rank}")
    print(f"World size: {world_size}")
    print(f"Address to bind to: {master_addr}")

    tcp_store = dist.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    dist.init_process_group(
        backend, rank=world_rank, world_size=world_size,
        store=tcp_store,
        timeout=timedelta(hours=2)
    )
    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    # NCCL errors can be caused by placing tensors on the wrong device
    # Oddly in our jobs we have 4 local nodes but each say cuda:0. Not sure
    # what's going on here...
    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            # Unfortunately, dist.broadcase will raise errors when using
            # the gloo distributed framework. This is related to how the two
            # frameworks copy tensors, but if you are seeing errors on this line,
            # just use nccl
            # p.detach()
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
