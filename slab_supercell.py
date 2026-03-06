import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
import math

def _validate_xy_scaling_matrix(scaling_matrix: ArrayLike) -> NDArray[np.int64]:
    """Validate and normalize a scaling matrix for in-plane slab supercells."""
    matrix = np.array(scaling_matrix)
    if matrix.shape == (3,):
        matrix = np.diag(matrix)
    elif matrix.shape != (3, 3):
        raise ValueError(
            "scaling_matrix must be a full 3x3 matrix or a sequence of three scaling factors."
        )

    matrix_int = np.rint(matrix).astype(np.int64)
    if not np.allclose(matrix, matrix_int):
        raise ValueError("scaling_matrix must contain only integer entries.")

    if not np.array_equal(matrix_int[2], np.array([0, 0, 1])):
        raise ValueError("The c lattice vector must remain unchanged: third row must be [0, 0, 1].")
    if not np.array_equal(matrix_int[:2, 2], np.array([0, 0])):
        raise ValueError("The a and b lattice vectors must not mix with c: first two rows must have 0 in column 3.")

    det = int(round(np.linalg.det(matrix_int)))
    if det == 0:
        raise ValueError("scaling_matrix must be invertible.")

    return matrix_int


def make_slab_xy_supercell_from_ouc(slab: Slab, scaling_matrix: ArrayLike) -> Slab:
    """Build an in-plane slab supercell by transforming the OUC, then rebuilding the slab.

    Args:
        slab (Slab): Input slab to transform.
        scaling_matrix (ArrayLike): Supercell scaling matrix. Accepts either:
            (1) a full 3x3 integer matrix, or (2) 3 scaling factors.
            The transformation is restricted to in-plane operations so that c
            remains unchanged.

    Returns:
        Slab: New slab supercell reconstructed from the transformed OUC.
    """
    matrix = _validate_xy_scaling_matrix(scaling_matrix)

    ouc = slab.oriented_unit_cell.copy()
    ouc.make_supercell(matrix)

    n_slab_layers = len(slab) / len(slab.oriented_unit_cell)
    if not math.isclose(n_slab_layers, round(n_slab_layers), abs_tol=1e-8):
        raise ValueError("Cannot infer an integer number of slab layers from the input slab and OUC.")
    n_slab_layers = int(round(n_slab_layers))

    n_total_layers = slab.lattice.c / slab.oriented_unit_cell.lattice.c
    if not math.isclose(n_total_layers, round(n_total_layers), abs_tol=1e-8):
        raise ValueError("Cannot infer an integer number of total layers from slab and OUC c lattice lengths.")
    n_total_layers = int(round(n_total_layers))

    if n_slab_layers <= 0 or n_total_layers < n_slab_layers:
        raise ValueError("Invalid slab/OUC layering inferred from input slab.")

    frac_coords = ouc.frac_coords + np.array([0, 0, -slab.shift])[None, :]
    frac_coords -= np.floor(frac_coords)
    frac_coords[:, 2] /= n_total_layers

    all_coords = []
    for idx in range(n_slab_layers):
        shifted_coords = frac_coords.copy()
        shifted_coords[:, 2] += idx / n_total_layers
        all_coords.extend(shifted_coords)

    site_props = {key: values * n_slab_layers for key, values in ouc.site_properties.items()}
    new_lattice = np.array(ouc.lattice.matrix)
    new_lattice[2] *= n_total_layers
    rebuilt = Structure(new_lattice, ouc.species_and_occu * n_slab_layers, all_coords, site_properties=site_props)

    scale_factor = np.dot(matrix, slab.scale_factor)
    new_slab = Slab(
        rebuilt.lattice,
        rebuilt.species_and_occu,
        rebuilt.frac_coords,
        slab.miller_index,
        ouc,
        slab.shift,
        scale_factor,
        reorient_lattice=slab.reorient_lattice,
        site_properties=rebuilt.site_properties,
        energy=slab.energy,
    )
    new_slab.reconstruction = slab.reconstruction
    return new_slab
