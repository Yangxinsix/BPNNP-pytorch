from ase.db import connect
import torch
from typing import List

def ase_properties(atoms):
    """Guess dataset format from an ASE atoms"""
    atoms_prop = {
        'elems': {'dtype':  'int32', 'shape': [None]},
        'coord': {'dtype':  'float', 'shape': [None, 3]}}

    if atoms.pbc.any():
        atoms_prop['cell'] = {'dtype': 'float', 'shape': [3, 3]}

    try:
        atoms.get_potential_energy()
        atoms_prop['energy'] = {'dtype': 'float', 'shape': []}
    except:
        pass

    try:
        atoms.get_forces()
        atoms_prop['forces'] = {'dtype': 'float', 'shape': [None, 3]}
    except:
        pass

    return atoms_prop

def ase_data_reader(atoms, atoms_prop):
    atoms_data = {
        'numbers': torch.tensor(atoms.get_global_number_of_atoms()),
        'elems': torch.tensor(atoms.numbers),
        'coord': torch.tensor(atoms.positions, dtype=torch.float),
    }
    if 'cell' in atoms_prop:
        atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)

    if 'energy' in atoms_prop:
        atoms_data['energy'] = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)

    if 'forces' in atoms_prop:
        atoms_data['forces'] = torch.tensor(atoms.get_forces(), dtype=torch.float)
    
    return atoms_data

class AseDataset(torch.utils.data.Dataset):
    def __init__(self, ase_db, cutoff, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(ase_db, str):
            self.db = connect(ase_db)
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_prop = ase_properties(self.db[1].toatoms())
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx+1].toatoms()    # ase database indexing from 1 
        atoms_data = ase_data_reader(atoms, self.atoms_prop)    
        return atoms_data
    
#def pad_and_stack(tensors: List[torch.Tensor]):
#    if tensors[0].shape:
#        return pad_sequence(
#            tensors, batch_first=True, padding_value=0
#        )
#    return torch.stack(tensors)

def cat_tensors(tensors: List[torch.Tensor]):
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True):
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
        
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items()}
    return collated