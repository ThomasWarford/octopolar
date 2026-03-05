from __future__ import annotations

from ase.visualize import view

from oxide_utils import (
    get_structure,
    make_slab,
    patch_slab_helpers,
    stabilize_slab_by_best_effort_move,
)


def view_structure(structure, viewer: str = "ase"):
    return view(structure.to_ase_atoms(), viewer=viewer)


def main(mp_id: str = "mp-1265"):
    patch_slab_helpers()

    structure = get_structure(mp_id).to_conventional()
    slab = make_slab(structure, miller_index=(1, 1, 1), min_slab_size=50, min_vacuum_size=10)
    solution = stabilize_slab_by_best_effort_move(slab)

    return {"slab": slab, "solution": solution}


if __name__ == "__main__":
    result = main()
    print(result["solution"])
