from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pulp
from joblib import Memory
from mp_api.client import MPRester
from pymatgen.core.surface import Slab, SlabGenerator


memory = Memory(".cachedir")


@memory.cache
def get_structure(mp_id: str):
    with MPRester() as mpr:
        return mpr.get_structure_by_material_id(mp_id)


def make_slab(
    structure,
    miller_index: Sequence[int] = (1, 1, 1),
    min_slab_size: float = 50.0,
    min_vacuum_size: float = 10.0,
    primitive: bool = True,
) -> Slab:
    slab_gen = SlabGenerator(
        structure,
        miller_index,
        min_slab_size,
        min_vacuum_size,
        primitive=primitive,
    )
    return slab_gen.get_slab()


def _get_dipole_density_per_ouc(slab: Slab) -> np.ndarray:
    ouc = slab.oriented_unit_cell.copy()
    ouc = ouc.add_oxidation_state_by_guess()
    centroid = np.sum(ouc.cart_coords, axis=0) / len(ouc)

    dipole = np.zeros(3)
    for site in ouc:
        charge = sum(
            getattr(sp, "oxi_state", 0) * amt for sp, amt in site.species.items()
        )
        dipole += charge * np.dot(site.coords - centroid, slab.normal) * slab.normal

    return dipole / ouc.volume


@property
def _required_surface_charge(slab: Slab) -> float:
    dipole = slab.get_dipole_density_per_ouc()
    return float(np.dot(dipole, slab.normal) * slab.surface_area)


def _ouc_site_to_slab_site_idx(slab: Slab, ouc_site_idx: int, bottom: bool = True) -> int:
    if bottom:
        return ouc_site_idx
    return len(slab) - len(slab.oriented_unit_cell) + ouc_site_idx


def _move_to_other_side(slab: Slab, ouc_site_idx: int, bottom: bool = True) -> None:
    site_idx = slab.ouc_site_to_slab_site_idx(ouc_site_idx, bottom=bottom)
    slab.sites[site_idx].coords += (
        (1 if bottom else -1) * slab.num_layers * slab.oriented_unit_cell.lattice.matrix[2]
    )


def patch_slab_helpers() -> None:
    Slab.get_dipole_density_per_ouc = _get_dipole_density_per_ouc
    Slab.required_surface_charge = _required_surface_charge
    Slab.ouc_site_to_slab_site_idx = _ouc_site_to_slab_site_idx
    Slab.move_to_other_side = _move_to_other_side


def solve_lexicographic_with_pulp(
    values: Sequence[float],
    target: float,
    depth_if_pos: Sequence[float],
    depth_if_neg: Sequence[float],
    can_be_pos: Sequence[bool],
    can_be_neg: Sequence[bool],
    msg: bool = False,
) -> dict:
    n = len(values)
    problem = pulp.LpProblem("surface_move_choice", pulp.LpMinimize)

    move_pos = [
        pulp.LpVariable(f"move_pos_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    ]
    move_neg = [
        pulp.LpVariable(f"move_neg_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    ]
    moved = [
        pulp.LpVariable(f"atom_moved_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    ]

    for i in range(n):
        if not can_be_pos[i]:
            problem += move_pos[i] == 0
        if not can_be_neg[i]:
            problem += move_neg[i] == 0
        problem += move_pos[i] + move_neg[i] <= 1
        problem += moved[i] == move_pos[i] + move_neg[i]

    problem += pulp.lpSum(values[i] * (move_pos[i] - move_neg[i]) for i in range(n)) == target

    solver = pulp.PULP_CBC_CMD(msg=msg)

    move_count = pulp.lpSum(moved)
    problem.objective = move_count
    problem.solve(solver)

    best_move_count = int(round(pulp.value(move_count)))
    problem += move_count == best_move_count

    depth_cost = pulp.lpSum(
        depth_if_pos[i] * move_pos[i] + depth_if_neg[i] * move_neg[i] for i in range(n)
    )
    problem.objective = depth_cost
    problem.solve(solver)

    weights = [
        int(round(pulp.value(move_pos[i]) - pulp.value(move_neg[i]))) for i in range(n)
    ]

    return {
        "weights": weights,
        "moved_count": sum(1 for w in weights if w != 0),
        "depth_cost": float(pulp.value(depth_cost)),
    }


def solve_best_effort_depth_first_with_pulp(
    values: Sequence[float],
    target: float,
    depth_if_pos: Sequence[float],
    depth_if_neg: Sequence[float],
    can_be_pos: Sequence[bool] | None = None,
    can_be_neg: Sequence[bool] | None = None,
    msg: bool = False,
    solver=None,
) -> dict:
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=msg)

    n = len(values)
    problem = pulp.LpProblem("best_effort_surface_move_choice", pulp.LpMinimize)

    choose_pos = [pulp.LpVariable(f"choose_pos_{i}", 0, 1, cat="Binary") for i in range(n)]
    choose_neg = [pulp.LpVariable(f"choose_neg_{i}", 0, 1, cat="Binary") for i in range(n)]

    for i in range(n):
        if (can_be_pos is not None) and (not can_be_pos[i]):
            problem += choose_pos[i] == 0
        if (can_be_neg is not None) and (not can_be_neg[i]):
            problem += choose_neg[i] == 0
        problem += choose_pos[i] + choose_neg[i] <= 1

    achieved_sum = pulp.lpSum(values[i] * (choose_pos[i] - choose_neg[i]) for i in range(n))

    residual_pos = pulp.LpVariable("residual_pos", lowBound=0)
    residual_neg = pulp.LpVariable("residual_neg", lowBound=0)
    problem += achieved_sum - target == residual_pos - residual_neg

    residual_abs = residual_pos + residual_neg
    depth_cost = pulp.lpSum(
        depth_if_pos[i] * choose_pos[i] + depth_if_neg[i] * choose_neg[i] for i in range(n)
    )

    problem.objective = residual_abs
    problem.solve(solver)

    best_residual_abs = float(pulp.value(residual_abs))
    problem += residual_abs == best_residual_abs

    problem.objective = depth_cost
    problem.solve(solver)

    weights = [
        int(round(pulp.value(choose_pos[i]) - pulp.value(choose_neg[i]))) for i in range(n)
    ]
    achieved = float(pulp.value(achieved_sum))
    residual = achieved - float(target)

    return {
        "weights": weights,
        "achieved_sum": achieved,
        "target": float(target),
        "residual": residual,
        "residual_abs": abs(residual),
        "depth_cost": float(pulp.value(depth_cost)),
    }


def process_uoc_linear_programming(slab: Slab) -> dict:
    uoc = slab.oriented_unit_cell.copy()
    np.testing.assert_allclose(
        uoc.lattice.matrix[:2],
        slab.lattice.matrix[:2],
        err_msg="Lattice mismatch between UOC and slab",
    )
    tol = 1e-6
    if np.linalg.norm(np.cross(uoc.lattice.matrix[2], slab.lattice.matrix[2])) >= tol:
        raise ValueError("Lattice mismatch between UOC and slab")

    distance_along_normal = np.dot(uoc.cart_coords, slab.normal)
    top_depth = max(distance_along_normal)
    bottom_depth = min(distance_along_normal)

    distance_from_top = top_depth - distance_along_normal
    distance_from_bottom = distance_along_normal - bottom_depth

    uoc.add_oxidation_state_by_guess()
    oxidation_states = [sp.oxi_state for sp in uoc.species]

    return {
        "distance_from_top": distance_from_top,
        "distance_from_bottom": distance_from_bottom,
        "oxidation_states": oxidation_states,
    }


def stabilize_slab_by_best_effort_move(slab: Slab, msg: bool = False) -> dict:
    slab_inputs = process_uoc_linear_programming(slab)
    solution = solve_best_effort_depth_first_with_pulp(
        values=slab_inputs["oxidation_states"],
        target=slab.required_surface_charge,
        depth_if_pos=slab_inputs["distance_from_top"],
        depth_if_neg=slab_inputs["distance_from_bottom"],
        msg=msg,
    )

    for ouc_site_idx, weight in enumerate(solution["weights"]):
        if weight == 1:
            slab.move_to_other_side(ouc_site_idx, bottom=False)
        elif weight == -1:
            slab.move_to_other_side(ouc_site_idx, bottom=True)

    return solution
