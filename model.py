import torch
from torch_scatter import scatter_min, scatter_add
from torch import nn
from typing import List
from collections import defaultdict
from ase.data import atomic_numbers

# Getting neighbor lists of a batch
class BatchNeighborList:
    def __init__(self, cutoff, compute_force=True):
        self.cutoff = cutoff
        self.compute_force = compute_force
        disp_mat = torch.zeros([3, 3, 3, 3]).long()
        helper_arr = torch.LongTensor([-1, 0, 1])
        disp_mat[:, :, :, 0] = torch.reshape(helper_arr, (3, 1, 1))
        disp_mat[:, :, :, 1] = torch.reshape(helper_arr, (1, 3, 1))
        disp_mat[:, :, :, 2] = torch.reshape(helper_arr, (1, 1, 3))
        self.disp_mat = disp_mat.reshape(27, 3)
        
    def __call__(self, tensors):
        # get batch cell, coordinates information and 
        # get atoms' image indices and real indices (where is the atom in the image)
        tensors['coord'].requires_grad_(self.compute_force)
        atom_cell = tensors['cell']
        atom_apos = tensors['coord']
        atom_counts = tensors['numbers']
        atom_image_ind = torch.arange(atom_counts.shape[0], device=tensors['coord'].device).repeat_interleave(atom_counts)
        image_atom_counts = torch.zeros_like(atom_counts)
        image_atom_counts[1:] = atom_counts[:-1]
        image_atom_counts = torch.cumsum(image_atom_counts, dim=0)
        atom_real_ind = image_atom_counts[atom_image_ind]
        
        # define cell parameter, cell shape, cell size
        c_len = torch.norm(atom_cell, dim=1).view(-1, 3)
        c_pos_shap = torch.div(c_len, self.cutoff, rounding_mode='floor').int()
        rc = c_len / c_pos_shap
        
        # define cell positions and cell indices
        cell_cpos = torch.ones(tuple(torch.max(c_pos_shap, dim=0)[0]), device=tensors['coord'].device)
        cell_cpos = cell_cpos.repeat(c_pos_shap.shape[0], 1, 1, 1)
        cind = torch.nonzero(cell_cpos)
        cell_cpos = cind[torch.all(cind[:, 1:] < c_pos_shap[cind[:, 0]], dim=1)]   # I'm incredible!!!
        cell_image_ind, cell_cpos = cell_cpos[:, 0], cell_cpos[:, 1:]
        
        # this matrix is used to calculate cell indices given cell postion vectors
        count_mat = torch.ones_like(c_pos_shap)
        count_mat[:, 0] = c_pos_shap[:, 1] * c_pos_shap[:, 2]
        count_mat[:, 1] = c_pos_shap[:, 2]
        
        # count how many cells there are for each images
        cell_counts = torch.unique_consecutive(cell_image_ind, return_counts=True)[1]
        image_cell_counts = torch.zeros_like(cell_counts)
        image_cell_counts[1:] = cell_counts[:-1]
        image_cell_counts = torch.cumsum(image_cell_counts, dim=0)
        
        # locate the positions of atoms in which cell
        atom_gind = torch.arange(atom_apos.size(0), device=tensors['coord'].device) + 1
        atom_cpos = torch.div(atom_apos, rc[atom_image_ind], rounding_mode='floor').long()
        atom_cind = torch.squeeze(torch.sum(atom_cpos * count_mat[atom_image_ind], dim=1))
        atom_cind += image_cell_counts[atom_image_ind]
        
        # get cell atom list
        atom_cind_sort, atom_cind_args = torch.sort(atom_cind, dim = 0, stable = True)
        cell_rind_min = scatter_min(atom_gind, atom_cind_sort)[0]
        atom_rind = atom_gind - torch.take(cell_rind_min, atom_cind_sort)
        atom_rpos = torch.stack((atom_cind_sort, atom_rind), dim = 1)
        cell_alsshap = torch.Size([cell_cpos.size(0), torch.max(atom_rind) + 1])
        cell_alst = torch.zeros(cell_alsshap, device=tensors['coord'].device).long()
        cell_alst = cell_alst.index_put(tuple(atom_rpos.t()), atom_cind_args + 1)
        
        # get cell neighbors and shifts to accurately calculate distance between atoms
        disp_mat = self.disp_mat.to(tensors['coord'].device)
        cell_npos = torch.unsqueeze(cell_cpos, 1) + disp_mat
        cell_nind = (cell_npos + c_pos_shap[cell_image_ind].unsqueeze(dim=1)) % c_pos_shap[cell_image_ind].unsqueeze(dim=1)
        cell_nind = torch.sum(cell_nind * count_mat[cell_image_ind].unsqueeze(dim=1), dim=-1)
        
        mask_1 = cell_npos < 0
        mask_2 = torch.ge(cell_npos, c_pos_shap[cell_image_ind].unsqueeze(dim=1))
        cell_nshift = torch.zeros_like(cell_npos).masked_fill_(mask_1, -1).masked_fill_(mask_2, 1)
        cell_nshift = cell_nshift * c_len[cell_image_ind].unsqueeze(dim=1)
        
        # get atoms' neighbor cells and atoms in these cells
        atom_cnind = cell_nind[atom_cind].squeeze() + image_cell_counts[atom_image_ind].unsqueeze(dim=-1)
        atom_cnshift = cell_nshift[atom_cind]
        atom_nind = cell_alst[atom_cnind]
        
        # calculate distances between atoms in neighboring cells
        pair_i_ind, neigh_ind, pair_j_ind = torch.where(atom_nind)
        # notice!! pair_j indices should be took out from atom_nind[pair_i_ind, neigh_ind, pair_j_ind], and minus 1!!!
        pair_j_ind = atom_nind[pair_i_ind, neigh_ind, pair_j_ind] - 1
        pair_shift = atom_cnshift[pair_i_ind, neigh_ind]
        pair_diff = (atom_apos[pair_j_ind] + pair_shift) - atom_apos[pair_i_ind]
        pair_dist = torch.norm(pair_diff, dim = 1)
        
        # screen pairs with distance smaller than cutoff radius
        ind_rc = torch.where((pair_dist > 0) & (pair_dist < self.cutoff))
        pair_i_aind = pair_i_ind[ind_rc]
        pair_j_aind = pair_j_ind[ind_rc]
        diff = pair_diff[ind_rc]
        dist = pair_dist[ind_rc]
        tensors['atom_image_idx'] = atom_image_ind   # image index of each atom
        tensors['pair_image_idx'] = atom_image_ind[pair_i_aind]   # image index of each pair
        tensors['atom_i_idx'] = pair_i_aind         # the index of center atom i in all atoms of this batch
        tensors['pair_i_idx'] = pair_i_aind - atom_real_ind[pair_i_aind]   # index of i in the image        
        tensors['n_indices'] = pair_j_aind - atom_real_ind[pair_j_aind]   # index of j in the image
        tensors['i_elems'] = tensors['elems'][pair_i_aind]   # element of pair i
        tensors['j_elems'] = tensors['elems'][pair_j_aind]   # element of pair j
        tensors['n_diff'] = diff                    # distance vector of R_j - R_i
        tensors['n_dist'] = dist                    # distance between pair i and j
        
        return tensors

def G2_SF(tensors, j_elem, eta, R_s, R_c):
    eta = eta.to(device=tensors['n_diff'].device)
    R_s = R_s.to(device=tensors['n_diff'].device)
    R_c = R_c.to(device=tensors['n_diff'].device)
    atom_i_idx = tensors['atom_i_idx']
#    pair_j_idx = tensors['pair_j_idx']
    counts = tensors['counts']
        
    # find the right neighbors
    j_mask = (tensors['j_elems'] == j_elem)
#    _, inverse_indices, counts = torch.unique_consecutive(pair_i_idx, return_inverse= True, return_counts=True)
    sfs = torch.zeros((counts.shape[0], R_c.shape[0]), device=tensors['n_diff'].device) 
    
    # calculate symmetry function values
    pair_dist = tensors['n_dist'][j_mask].unsqueeze(dim=-1)
    # debug
    pair_fc = 0.5 * (torch.cos(pair_dist * torch.pi / R_c) + 1)
    pair_sfs = torch.exp(-eta * (pair_dist - R_s) ** 2) * pair_fc
    sfs.index_add_(0, atom_i_idx[j_mask], pair_sfs)
    
    return sfs

def G3_SF(tensors, j_elem, kappa, R_c):
    kappa = kappa.to(device=tensors['n_diff'].device)
    R_c = R_c.to(device=tensors['n_diff'].device)
    atom_i_idx = tensors['atom_i_idx']
    pair_j_idx = tensors['pair_j_idx']
    counts = tensors['counts']
        
    # find the right neighbors
    j_mask = (tensors['j_elems'] == j_elem)
#    _, inverse_indices, counts = torch.unique_consecutive(pair_i_idx, return_inverse= True, return_counts=True)
    sfs = torch.zeros((counts.shape[0], R_c.shape[0]), device=tensors['n_diff'].device)
    
    # calculate symmetry function values
    pair_dist = tensors['n_dist'][j_mask].unsqueeze(dim=-1)
    pair_fc = 0.5 * (torch.cos(pair_dist * torch.pi / R_c) + 1)
    pair_sfs = torch.cos(kappa * pair_dist) * pair_fc
    sfs.index_add_(0, atom_i_idx[j_mask], pair_sfs)
    
    return sfs

def G5_SF(tensors, j_elem, k_elem, zeta, Lambda, eta, R_c):
    zeta = zeta.to(device=tensors['n_diff'].device)
    Lambda = Lambda.to(device=tensors['n_diff'].device)
    eta = eta.to(device=tensors['n_diff'].device)
    R_c = R_c.to(device=tensors['n_diff'].device)
    diff = tensors['n_diff']
    dist = tensors['n_dist']   
    atom_i_idx = tensors['atom_i_idx']
    counts = tensors['counts']
    
    j_mask = (tensors['j_elems'] == j_elem)
    k_mask = (tensors['j_elems'] == k_elem)
    
    # get relative index of neighbors
    atom_idx_j_masked, j_inv_idx, j_counts = torch.unique_consecutive(
        atom_i_idx[j_mask], return_inverse=True, return_counts=True,
    )
    atom_idx_k_masked, k_inv_idx, k_counts = torch.unique_consecutive(
        atom_i_idx[k_mask], return_inverse=True, return_counts=True,
    )
    
    g_idx = torch.arange(j_inv_idx.shape[0], device=tensors['n_diff'].device)
    idx_min, _ = scatter_min(g_idx, j_inv_idx)
    j_ridx = g_idx - idx_min[j_inv_idx]

    g_idx = torch.arange(k_inv_idx.shape[0], device=tensors['n_diff'].device)
    idx_min, _ = scatter_min(g_idx, k_inv_idx)
    k_ridx = g_idx - idx_min[k_inv_idx]
    
    # get the matrix of M * N * ..., 
    # where M is the number of center atoms, 
    # N is the maximum number of their j or k neighbors
    diff_ij = torch.zeros((counts.shape[0], j_counts.max(), 3), device=tensors['n_diff'].device)
    diff_ik = torch.zeros((counts.shape[0], k_counts.max(), 3), device=tensors['n_diff'].device)
    dist_ij = torch.zeros((counts.shape[0], j_counts.max()), device=tensors['n_diff'].device)
    dist_ik = torch.zeros((counts.shape[0], k_counts.max()), device=tensors['n_diff'].device)
    
    diff_ij[atom_i_idx[j_mask], j_ridx] = diff[j_mask]
    diff_ik[atom_i_idx[k_mask], k_ridx] = diff[k_mask]
    dist_ij[atom_i_idx[j_mask], j_ridx] = dist[j_mask]
    dist_ik[atom_i_idx[k_mask], k_ridx] = dist[k_mask]
    
    # calculate the values of different parts in angular symmetry functions
    diff_ijk = torch.einsum("ijk, ilk -> ijl", diff_ij, diff_ik)
    dist_prod = (dist_ij.unsqueeze(dim=-1) * dist_ik.unsqueeze(dim=-2))
    
    # handling situation that j = k
    if j_elem == k_elem:
 #       dist_prod = torch.triu(dist_prod, diagonal = 1) + torch.tril(dist_prod, diagonal = -1)
        dist_prod = torch.triu(dist_prod, diagonal = 1)
        
    idx_i, idx_j, idx_k = torch.where(dist_prod) 

    part_1 = diff_ijk[idx_i, idx_j, idx_k] / dist_prod[idx_i, idx_j, idx_k]
    part_1 = (part_1.unsqueeze(dim=1) * Lambda + 1) ** zeta

    part_2 = torch.exp(-eta * (dist_ij[idx_i, idx_j] ** 2 + dist_ik[idx_i, idx_k] ** 2).unsqueeze(dim=-1))

    pair_fc_ij = 0.5 * (torch.cos(torch.pi * dist_ij[idx_i, idx_j].unsqueeze(dim=-1) / R_c) + 1)
    pair_fc_ik = 0.5 * (torch.cos(torch.pi * dist_ik[idx_i, idx_k].unsqueeze(dim=-1) / R_c) + 1)   
    part_3 = pair_fc_ij * pair_fc_ik

    sfs = torch.zeros((counts.shape[0], R_c.shape[0]), device=tensors['n_diff'].device)
    sfs = sfs.index_add(0, idx_i, part_1 * part_2 * part_3 * 2 ** (1-zeta))
    
    return sfs

def G4_SF(tensors, j_elem, k_elem, zeta, Lambda, eta, R_c):
    zeta = zeta.to(device=tensors['n_diff'].device)
    Lambda = Lambda.to(device=tensors['n_diff'].device)
    eta = eta.to(device=tensors['n_diff'].device)
    R_c = R_c.to(device=tensors['n_diff'].device)
    diff = tensors['n_diff']
    dist = tensors['n_dist']   
    atom_i_idx = tensors['atom_i_idx']
    counts = tensors['counts']
    
    j_mask = (tensors['j_elems'] == j_elem)
    k_mask = (tensors['j_elems'] == k_elem)
    
    # get relative index of neighbors
    atom_idx_j_masked, j_inv_idx, j_counts = torch.unique_consecutive(
        atom_i_idx[j_mask], return_inverse=True, return_counts=True,
    )
    atom_idx_k_masked, k_inv_idx, k_counts = torch.unique_consecutive(
        atom_i_idx[k_mask], return_inverse=True, return_counts=True,
    )
    
    g_idx = torch.arange(j_inv_idx.shape[0], device=tensors['n_diff'].device)
    idx_min, _ = scatter_min(g_idx, j_inv_idx)
    j_ridx = g_idx - idx_min[j_inv_idx]

    g_idx = torch.arange(k_inv_idx.shape[0], device=tensors['n_diff'].device)
    idx_min, _ = scatter_min(g_idx, k_inv_idx)
    k_ridx = g_idx - idx_min[k_inv_idx]
    
    # get the matrix of M * N * ..., 
    # where M is the number of center atoms, 
    # N is the maximum number of their j or k neighbors
    diff_ij = torch.zeros((counts.shape[0], j_counts.max(), 3), device=tensors['n_diff'].device)
    diff_ik = torch.zeros((counts.shape[0], k_counts.max(), 3), device=tensors['n_diff'].device)
    dist_ij = torch.zeros((counts.shape[0], j_counts.max()), device=tensors['n_diff'].device)
    dist_ik = torch.zeros((counts.shape[0], k_counts.max()), device=tensors['n_diff'].device)
    
    diff_ij[atom_i_idx[j_mask], j_ridx] = diff[j_mask]
    diff_ik[atom_i_idx[k_mask], k_ridx] = diff[k_mask]
    dist_ij[atom_i_idx[j_mask], j_ridx] = dist[j_mask]
    dist_ik[atom_i_idx[k_mask], k_ridx] = dist[k_mask]
    
    # calculate the values of different parts in angular symmetry functions
    diff_ijk = torch.einsum("ijk, ilk -> ijl", diff_ij, diff_ik)
    dist_prod = (dist_ij.unsqueeze(dim=-1) * dist_ik.unsqueeze(dim=-2))
    
    # handling situation that j = k
    if j_elem == k_elem:
#        dist_prod = torch.triu(dist_prod, diagonal = 1) + torch.tril(dist_prod, diagonal = -1)
        dist_prod = torch.triu(dist_prod, diagonal = 1)
        
    idx_i, idx_j, idx_k = torch.where(dist_prod)
    pair_dist_jk = torch.norm(diff_ik[idx_i, idx_k] - diff_ij[idx_i, idx_j], dim=-1)
    jk_mask = pair_dist_jk.unsqueeze(-1) < R_c[0]
    
    part_1 = diff_ijk[idx_i, idx_j, idx_k] / dist_prod[idx_i, idx_j, idx_k]
    part_1 = (part_1.unsqueeze(dim=1) * Lambda + 1) ** zeta

    part_2 = torch.exp(-eta * (dist_ij[idx_i, idx_j] ** 2 + dist_ik[idx_i, idx_k] ** 2).unsqueeze(dim=-1))

    pair_fc_ij = 0.5 * (torch.cos(torch.pi * dist_ij[idx_i, idx_j].unsqueeze(dim=-1) / R_c) + 1)
    pair_fc_ik = 0.5 * (torch.cos(torch.pi * dist_ik[idx_i, idx_k].unsqueeze(dim=-1) / R_c) + 1)  
    pair_fc_jk = 0.5 * (torch.cos(torch.pi * pair_dist_jk.unsqueeze(dim=-1) / R_c) + 1)  
    part_3 = pair_fc_ij * pair_fc_ik * pair_fc_jk

    sfs = torch.zeros((counts.shape[0], R_c.shape[0]), device=tensors['n_diff'].device)
    sfs = sfs.index_add(0, idx_i[jk_mask], (part_1 * part_2 * part_3 * 2 ** (1-zeta))[jk_mask])
    
    return sfs

bp_sf_fns = {'G2': G2_SF, 'G3': G3_SF, 'G4': G4_SF, 'G5': G5_SF}
class BPSymmFunc:
    """
    Get Behler-Parrinello style symmetry function values of atoms
    """
    def __init__(self, sf_spec):
        self.sf_spec = defaultdict(list)
        for elem, elem_spec in sf_spec.items():
            for spec in elem_spec:
                fn = bp_sf_fns[spec['type']]
                options = {k:torch.FloatTensor(v)
                           if isinstance(v, list) else v 
                           for k, v in spec.items() if k != 'type'}
                self.sf_spec[elem].append((fn, options))
    
    def __call__(self, tensors):
        fps = {}
        for elem, elem_spec in self.sf_spec.items():    
            sfs = []
            i_elem = atomic_numbers[elem]
            i_masked = self.get_i_masked(tensors, i_elem=i_elem)
            for fn, options in elem_spec:
                sf = fn(i_masked, **options)
                sfs.append(sf)
            sfs =  torch.hstack(sfs)
            fps['{}_sfs'.format(elem)] = sfs
            fps['{}_image_idx'.format(elem)] = i_masked['atom_image_idx']
            
        tensors['fps'] = fps
        return tensors
    
    def get_i_masked(self, tensors, i_elem):
        """This function aims to construct M * N matrices, 
        where M is the number of selected center atoms,
        N is the maximum number of these atoms' neighbors.
        """
        i_mask = (tensors['i_elems'] == i_elem)
        pair_image_idx = tensors['pair_image_idx']
        pair_i_idx = tensors['pair_i_idx']
        atom_image_idx = tensors['atom_image_idx'][tensors['elems'] == i_elem]
        numbers = tensors['numbers']
        _, inverse_indices, counts = torch.unique_consecutive(pair_i_idx[i_mask],
                                                              return_inverse=True,
                                                              return_counts=True)
    
        i_masked = {
            'atom_image_idx': atom_image_idx,
            'atom_i_idx': inverse_indices,             # atom indices of i masked pairs in this batch           
            'n_dist': tensors['n_dist'][i_mask],       # distances of i masked pairs
            'n_diff': tensors['n_diff'][i_mask],       # distance vectors of i masked pairs
            'counts': counts,                          # how many pairs there are for each i atom
            'j_elems': tensors['j_elems'][i_mask],     # j elements of pairs
        }
        return i_masked

class BPNNP(nn.Module):
    def __init__(self, sf_spec, layer_size: list, cutoff, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = BatchNeighborList(cutoff)
        self.sf_spec = sf_spec
        self.fingerprint = BPSymmFunc(self.sf_spec)
        self.elems_num = len(self.sf_spec)
        hidden_layers = [nn.Sequential(nn.Linear(layer_size[i],
                                                 layer_size[i+1]),
                                       nn.BatchNorm1d(layer_size[i+1]),   # batch normalization
                                       nn.Sigmoid(),
                                      )
                         for i in range(len(layer_size) - 1)]
        self.elem_layers = {}
        for elem, elem_specs in self.sf_spec.items():
            input_size = 0
            for spec in elem_specs:
                input_size += len(spec['R_c'])
                
            input_layer = nn.Sequential(nn.Linear(input_size, layer_size[0]),  
                                        nn.BatchNorm1d(layer_size[0]),     # batch normalization
                                        nn.Sigmoid(),
                                       )    
            output_layer = nn.Linear(layer_size[-1], 1)
            layers = [input_layer] + hidden_layers + [output_layer]
            self.elem_layers[elem] = nn.ModuleList(layers)
        self.elem_layers = nn.ModuleDict(self.elem_layers)
        self.scale = scale

    def forward(self, tensors):
        """The input is a dict of tensors, which is directed obtained from Dataset 
        with n_indices, n_diff, and n_dist keys"""
        tensors = self.preprocess(tensors)
        tensors = self.fingerprint(tensors)
        energy = []
        image_idx = []
        for k, layers in self.elem_layers.items():
            x = tensors['fps'][k + '_sfs']
            for layer in layers:
                x = layer(x)
                
            energy.append(x)
            image_idx.append(tensors['fps'][k + '_image_idx'])
        
        energy = torch.cat(energy).squeeze(dim=-1)
        image_idx = torch.cat(image_idx)
        dE_dxyz = torch.autograd.grad(
            energy,
            tensors['coord'],
            grad_outputs=torch.ones_like(energy),
            retain_graph=True,
            create_graph=True,
        )[0]
        energy = scatter_add(energy, image_idx)
        result_dict = {'energy': energy}
        forces = -dE_dxyz       
        result_dict['forces'] = forces
        return result_dict