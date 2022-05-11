import math
import numpy as np
import torch
import re, os, logging, json, time, argparse

from ase.data import atomic_numbers
from ase.db import connect

from data import AseDataset, collate_atomsdata
from model import BPNNP

def split_data(dataset):
    # Load or generate splits
#    if args.split_file:
#        with open(args.split_file, "r") as fp:
#            splits = json.load(fp)
#    else:
#        datalen = len(dataset)
#        num_validation = int(math.ceil(datalen * 0.10))
#        indices = np.random.permutation(len(dataset))
#        splits = {
#            "train": indices[num_validation:].tolist(),
#            "validation": indices[:num_validation].tolist(),
#        }

    datalen = len(dataset)
    num_validation = int(math.ceil(datalen * 0.10))
    indices = np.random.permutation(len(dataset))
    splits = {
        "train": indices[num_validation:].tolist(),
        "validation": indices[:num_validation].tolist(),
    }

    # Save split file
#    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
    with open("datasplits.json", "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits

n2p2_sf_type = {'2': 'G2', '3': 'G4', '9': 'G5'}
def get_sf_dict(input_file = 'input.nn'):
    """ Read symmetry functions parameters from input.nn """
    lines = []
    for line in open(input_file, 'r'):
        if line.startswith('symfunc'):
            lines.append(line.strip())
    sf_specs = {}
    for line in lines:
        line = line.split()
        if re.match('[a-zA-Z]', line[4]):
            try:
                sf_specs[" ".join(line[1:5])]
            except:
                sf_specs[" ".join(line[1:5])] = {'eta': [], 'Lambda': [], 'zeta': [], 'R_c': []}
            finally:
                sf_specs[" ".join(line[1:5])]['eta'].append(float(line[5]))
                sf_specs[" ".join(line[1:5])]['Lambda'].append(float(line[6]))
                sf_specs[" ".join(line[1:5])]['zeta'].append(float(line[7]))
                sf_specs[" ".join(line[1:5])]['R_c'].append(float(line[8]))
            
        else:
            try:
                sf_specs[" ".join(line[1:4])]
            except:
                sf_specs[" ".join(line[1:4])] = {'eta': [], 'R_s': [], 'R_c': []}
            finally:
                sf_specs[" ".join(line[1:4])]['eta'].append(float(line[4]))
                sf_specs[" ".join(line[1:4])]['R_s'].append(float(line[5]))
                sf_specs[" ".join(line[1:4])]['R_c'].append(float(line[6]))

    new_sf_specs = {}
    for k1, v1 in sf_specs.items():
        sf_type = k1.split()
        sf_dict = {'type': n2p2_sf_type[sf_type[1]], 'j_elem': atomic_numbers[sf_type[2]]}
        sf_dict.update({k2: v2 for k2, v2 in v1.items()})
        if len(sf_type) == 4:
            sf_dict.update({'k_elem': atomic_numbers[sf_type[3]]})
        try:
            new_sf_specs[sf_type[0]]
        except:
            new_sf_specs[sf_type[0]] = []
        finally:
            new_sf_specs[sf_type[0]].append(sf_dict)
            
    return new_sf_specs

def forces_criterion(predicted, target, reduction="mean"):
    # predicted, target are (bs, max_nodes, 3) tensors
    # node_count is (bs) tensor
    diff = predicted - target
    total_squared_norm = torch.norm(diff, dim=1)  # bs
    if reduction == "mean":
        scalar = torch.mean(total_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar

# def eval_model(net, dataloader, device):
#     energy_ae = 0
#     energy_se = 0
#     forces_ae = 0
#     forces_se = 0
#     running_counts = 0

#     for device_batch in dataloader:
#         batch = {
#             k: v.to(device=device, non_blocking=True) for k, v in device_batch.items()
#         }
#         output = net(batch)
#         running_counts += batch['energy'].shape[0]
#         energy_ae += torch.abs(output['energy'] - batch['energy'])
#         energy_ae = torch.abs()
        


# Setup training log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),
        logging.StreamHandler(),
    ],
)

# Get dataset and split data
ase_data = 'dft_structures.db'
logging.info("loading data %s", ase_data)
dataset = AseDataset(connect(ase_data), cutoff=6.0)
datasplits = split_data(dataset=dataset)

# Setup loaders
train_loader = torch.utils.data.DataLoader(
    datasplits["train"],
    batch_size=32,
    sampler=torch.utils.data.RandomSampler(datasplits["train"]),
    collate_fn=collate_atomsdata,
    )
val_loader = torch.utils.data.DataLoader(
    datasplits["validation"], 
    batch_size=32, 
    collate_fn=collate_atomsdata,
)

# Initialize model
sf_specs = get_sf_dict()
net = BPNNP(sf_specs, layer_size=[30, 30], cutoff=6.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Setup optimizer
initial_lr = 0.0001    # learning rate
optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
criterion = torch.nn.MSELoss(reduction="mean")
scheduler_fn = lambda step: 0.96 ** (step / 100000)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

# Training
forces_weight=0.90
step = 0
start = time.time()
epoch = 100
logging.info("start training")
for i in range(epoch):
    for device_batch in train_loader:    
        batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in device_batch.items()
        }
        optimizer.zero_grad()

        # Forward, backward and optimize
        outputs = net(batch)
        energy_loss = criterion(outputs["energy"], batch["energy"])
        if forces_weight:
            forces_loss = forces_criterion(outputs["forces"], batch["forces"], reduction="mean")
        else:
            forces_loss = 0.0
        total_loss = (
            forces_weight * forces_loss
            + (1 - forces_weight) * energy_loss
        )
        
        total_loss.backward()
        optimizer.step()
        step += 1
        if (step % 100 == 0): 
            end = time.time()
            cost = end - start
            start = time.time()
            logging.info(
                    "Timecost: %g, step: %d, total loss: %g, forces loss: %g",
                    cost,
                    step,
                    total_loss,
                    energy_loss,
                    forces_loss,
                )