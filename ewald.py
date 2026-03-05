import torch
from pathlib import Path
from scipy.constants import pi
from ase.io import read
import numpy as np

from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.gto_utils import gto_basis_kspace_cutoff

from pymatgen.core import Structure


torch.set_default_dtype(torch.float64)

# density_max_l is the multipole order. 0=charges, 1=charges+dipoles, etc...
# the realspace evalation is currently supported for only l<=1. 
density_max_l = 0

# this is sigma_n in the GTO basis, we generally use quite wide gaussians in ML models
density_smearing_width = 0.25

# use this function for a heuristic estimate of the k-space cutoff, 
# which is needed to determine the number of k-vectors to use in the reciprocal space sum.
KSPACE_CUTOFF = 3 * gto_basis_kspace_cutoff(
    sigmas=[density_smearing_width],
    max_l=density_max_l,
)

ENERGY_BLOCK = GTOElectrostaticEnergy(
    density_max_l=density_max_l,
    density_smearing_width=density_smearing_width,
    kspace_cutoff=KSPACE_CUTOFF,
    include_self_interaction=False,
)


def atoms_list_to_batch(atoms_list):
    positions = []
    batch = []
    cells = []
    volumes = []
    pbcs = []

    for graph_i, atoms in enumerate(atoms_list):
        pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype())
        positions.append(pos)
        batch.append(torch.full((pos.shape[0],), graph_i, dtype=torch.long))

        cell = torch.tensor(atoms.cell.array, dtype=torch.get_default_dtype())
        if torch.allclose(cell, torch.zeros_like(cell)):
            cell = torch.eye(3, dtype=cell.dtype)
        cells.append(cell)
        volumes.append(abs(torch.det(cell)))
        pbcs.append(torch.tensor(atoms.pbc, dtype=torch.bool))

    return (
        torch.cat(positions, dim=0),
        torch.cat(batch, dim=0),
        torch.stack(cells, dim=0),
        torch.stack(volumes, dim=0),
        torch.stack(pbcs, dim=0),
    )

def get_ewald_energy(structure: Structure, pbc: list[bool]):
    atoms = structure.to_ase_atoms()
    atoms.set_pbc(pbc)

    multipoles = atoms.arrays['oxi_states']
    multipoles = torch.tensor(multipoles).view(-1, 1)


    positions, batch, cell, volume, pbc = atoms_list_to_batch([atoms])

    r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=KSPACE_CUTOFF,
        cell_vectors=cell,
        r_cell_vectors=r_cell,
)

    energy = ENERGY_BLOCK(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=multipoles,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )

    return energy