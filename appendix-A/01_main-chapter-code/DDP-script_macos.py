import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


def setup_ddp(rank, world_size):
    """
    Setup for Distributed Data Parallel
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group using gloo backend
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )


def prepare_dataloader(rank, world_size):
    # Create sample data
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
        [-1.1, 3.0],
        [-0.8, 2.8],
        [2.5, -1.3]
    ], dtype=torch.float32)
    
    y_train = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1], dtype=torch.long)
    
    # Create dataset
    train_dataset = ToyDataset(X_train, y_train)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        sampler=sampler,
        drop_last=True
    )
    
    return train_loader


def cleanup():
    """
    Clean up distributed training
    """
    destroy_process_group()


def train(rank, world_size, num_epochs):
    print(f"Running training on rank {rank}")
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Prepare dataloader
    train_loader = prepare_dataloader(rank, world_size)
    
    # Create model and keep it on CPU for DDP
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    
    # Initialize DDP model on CPU
    model = DDP(model)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Check if MPS is available for data processing
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Rank {rank} using MPS device for data processing")
    else:
        device = torch.device("cpu")
        print(f"Rank {rank} using CPU device")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device (MPS or CPU)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Move data back to CPU for forward pass
            output = model(data.cpu())
            
            # Move output to device for loss calculation
            output = output.to(device)
            loss = F.cross_entropy(output, target)
            
            # Move loss to CPU for backward pass
            loss.cpu().backward()
            
            optimizer.step()
            
            if batch_idx % 2 == 0:
                print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
    
    cleanup()


def run_distributed_training(world_size, num_epochs):
    """
    Spawn processes for distributed training
    """
    mp.spawn(
        train,
        args=(world_size, num_epochs),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Print PyTorch and device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    WORLD_SIZE = 8  # Number of processes
    NUM_EPOCHS = 5
    
    # Run distributed training
    run_distributed_training(WORLD_SIZE, NUM_EPOCHS)