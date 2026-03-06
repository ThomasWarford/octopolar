import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab


def make_slab_from_ouc(
    n_layers: int,
    vacuum: float,
    ouc: Structure,
    *,
    miller_index: tuple[int, int, int] = (0, 0, 1),
    shift: float = 0.0,
    scale_factor: ArrayLike = (1, 1, 1),
    reorient_lattice: bool = True,
    energy: float | None = None,
    reconstruction: str | None = None,
) -> Slab:
    """Build a slab by stacking an OUC along c and adding vacuum.

    Args:
        n_layers: Number of OUC repeats in the slab region.
        vacuum: Vacuum thickness in angstrom.
        ouc: Oriented unit cell to stack.
        miller_index: Slab Miller index metadata for the returned Slab.
        shift: Slab shift metadata for the returned Slab.
        scale_factor: Scale factor metadata for the returned Slab.
        reorient_lattice: Passed through to Slab constructor.
        energy: Optional slab energy metadata.
        reconstruction: Optional reconstruction metadata.

    Returns:
        Slab: A slab built from the provided OUC.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be a positive integer.")
    if vacuum < 0:
        raise ValueError("vacuum must be non-negative.")

    ouc = ouc.copy()
    c_length = ouc.lattice.c

    c_scale = n_layers + vacuum / c_length
    new_lattice = np.array(ouc.lattice.matrix)
    new_lattice[2] *= c_scale

    all_cart_coords = []
    for idx in range(n_layers):
        all_cart_coords.extend(ouc.cart_coords + idx * ouc.lattice.matrix[2])

    rebuilt = Structure(
        new_lattice,
        ouc.species_and_occu * n_layers,
        all_cart_coords,
        coords_are_cartesian=True,
        site_properties={key: values * n_layers for key, values in ouc.site_properties.items()},
    )

    slab = Slab(
        rebuilt.lattice,
        rebuilt.species_and_occu,
        rebuilt.frac_coords,
        miller_index,
        ouc,
        shift,
        scale_factor,
        reorient_lattice=reorient_lattice,
        site_properties=rebuilt.site_properties,
        energy=energy,
    )
    slab.reconstruction = reconstruction
    return slab

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

    slab_thickness = n_slab_layers * slab.oriented_unit_cell.lattice.c
    vacuum = slab.lattice.c - slab_thickness
    if n_slab_layers <= 0 or vacuum < -1e-8:
        raise ValueError("Invalid slab/OUC layering inferred from input slab.")
    vacuum = max(0.0, vacuum)

    scale_factor = np.dot(matrix, np.array(slab.scale_factor))
    new_slab = make_slab_from_ouc(
        n_layers=n_slab_layers,
        vacuum=vacuum,
        ouc=ouc,
        miller_index=slab.miller_index,
        shift=slab.shift,
        scale_factor=scale_factor,
        reorient_lattice=slab.reorient_lattice,
        energy=slab.energy,
        reconstruction=slab.reconstruction,
    )
    return new_slab
