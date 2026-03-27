#!/usr/bin/env python3
"""Compare two CIF crystal structures from geometric and physical perspectives."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass
class BasicMetrics:
    formula: str
    n_sites: int
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float
    density: float
    spacegroup_symbol: str
    spacegroup_number: int
    avg_nearest_neighbor: float
    avg_coordination_within_3a: float
    packing_fraction: float
    avg_electronegativity: float
    avg_atomic_mass: float
    avg_valence_electrons: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较两个 CIF 文件在几何与物理层面的差异（晶胞、对称性、局域环境与统计物性）"
    )
    parser.add_argument("cif1", type=Path, help="第一个 CIF 文件路径")
    parser.add_argument("cif2", type=Path, help="第二个 CIF 文件路径")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="先将结构标准化为常规晶胞再比较（建议用于不同设置的同构结构）",
    )
    parser.add_argument(
        "--primitive",
        action="store_true",
        help="先转为原胞再比较",
    )
    parser.add_argument(
        "--rdf-cutoff",
        type=float,
        default=10.0,
        help="RDF 统计截断半径（Å），默认 10.0",
    )
    parser.add_argument(
        "--rdf-bin",
        type=float,
        default=0.05,
        help="RDF 直方图 bin 宽度（Å），默认 0.05",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选：把结果写入 JSON 文件",
    )
    return parser.parse_args()


def preprocess_structure(structure: Structure, standardize: bool, primitive: bool) -> Structure:
    s = structure.copy()
    if standardize:
        s = SpacegroupAnalyzer(s, symprec=1e-2).get_conventional_standard_structure()
    if primitive:
        s = SpacegroupAnalyzer(s, symprec=1e-2).get_primitive_standard_structure()
    return s


def nearest_neighbor_distances(structure: Structure) -> np.ndarray:
    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    return dm.min(axis=1)


def avg_coordination_within_cutoff(structure: Structure, cutoff: float = 3.0) -> float:
    neighbor_counts = [len(structure.get_neighbors(site, cutoff)) for site in structure.sites]
    return float(np.mean(neighbor_counts)) if neighbor_counts else 0.0


def packing_fraction(structure: Structure) -> float:
    sphere_volume = 0.0
    for site in structure.sites:
        e = Element(site.specie.symbol)
        r = float(e.atomic_radius or 0.0)
        sphere_volume += (4.0 / 3.0) * math.pi * r**3
    if structure.volume <= 0:
        return 0.0
    return sphere_volume / structure.volume


def composition_physical_descriptors(structure: Structure) -> tuple[float, float, float]:
    comp = structure.composition
    total_atoms = sum(comp.values())
    if total_atoms == 0:
        return 0.0, 0.0, 0.0

    avg_x = 0.0
    avg_mass = 0.0
    avg_valence = 0.0

    for el, amount in comp.get_el_amt_dict().items():
        e = Element(el)
        weight = amount / total_atoms
        avg_x += weight * float(e.X or 0.0)
        avg_mass += weight * float(e.atomic_mass)
        valence = float(e.group) if e.group is not None else 0.0
        avg_valence += weight * valence

    return avg_x, avg_mass, avg_valence


def summarize_structure(structure: Structure) -> BasicMetrics:
    sga = SpacegroupAnalyzer(structure, symprec=1e-2)
    nn = nearest_neighbor_distances(structure)
    avg_x, avg_mass, avg_valence = composition_physical_descriptors(structure)

    return BasicMetrics(
        formula=structure.composition.reduced_formula,
        n_sites=len(structure.sites),
        a=float(structure.lattice.a),
        b=float(structure.lattice.b),
        c=float(structure.lattice.c),
        alpha=float(structure.lattice.alpha),
        beta=float(structure.lattice.beta),
        gamma=float(structure.lattice.gamma),
        volume=float(structure.volume),
        density=float(structure.density),
        spacegroup_symbol=str(sga.get_space_group_symbol()),
        spacegroup_number=int(sga.get_space_group_number()),
        avg_nearest_neighbor=float(np.mean(nn)),
        avg_coordination_within_3a=avg_coordination_within_cutoff(structure, 3.0),
        packing_fraction=packing_fraction(structure),
        avg_electronegativity=avg_x,
        avg_atomic_mass=avg_mass,
        avg_valence_electrons=avg_valence,
    )


def pair_distance_histogram(structure: Structure, cutoff: float, bin_width: float) -> np.ndarray:
    dm = structure.distance_matrix
    iu = np.triu_indices_from(dm, k=1)
    distances = dm[iu]
    distances = distances[(distances > 1e-8) & (distances <= cutoff)]
    n_bins = max(1, int(cutoff / bin_width))
    hist, _ = np.histogram(distances, bins=n_bins, range=(0.0, cutoff), density=True)
    return hist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def relative_delta(v1: float, v2: float) -> float:
    if abs(v1) < 1e-12:
        return float("inf") if abs(v2) > 1e-12 else 0.0
    return (v2 - v1) / v1 * 100.0


def compare_structures(
    s1: Structure,
    s2: Structure,
    rdf_cutoff: float,
    rdf_bin: float,
) -> dict[str, Any]:
    m1 = summarize_structure(s1)
    m2 = summarize_structure(s2)

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    is_same_framework = matcher.fit(s1, s2)
    rms_result = matcher.get_rms_dist(s1, s2)
    rms = float(rms_result[0]) if rms_result else None

    h1 = pair_distance_histogram(s1, rdf_cutoff, rdf_bin)
    h2 = pair_distance_histogram(s2, rdf_cutoff, rdf_bin)

    geometry_comparison = {
        "volume_delta_percent": relative_delta(m1.volume, m2.volume),
        "density_delta_percent": relative_delta(m1.density, m2.density),
        "lattice_delta_percent": {
            "a": relative_delta(m1.a, m2.a),
            "b": relative_delta(m1.b, m2.b),
            "c": relative_delta(m1.c, m2.c),
        },
        "angle_delta_deg": {
            "alpha": m2.alpha - m1.alpha,
            "beta": m2.beta - m1.beta,
            "gamma": m2.gamma - m1.gamma,
        },
        "avg_nearest_neighbor_delta_percent": relative_delta(
            m1.avg_nearest_neighbor, m2.avg_nearest_neighbor
        ),
        "coordination_delta_percent": relative_delta(
            m1.avg_coordination_within_3a, m2.avg_coordination_within_3a
        ),
        "rdf_cosine_similarity": cosine_similarity(h1, h2),
        "structure_matcher_fit": is_same_framework,
        "structure_matcher_rms": rms,
    }

    physical_comparison = {
        "formula_equal": m1.formula == m2.formula,
        "spacegroup_equal": (m1.spacegroup_number == m2.spacegroup_number),
        "spacegroup_change": {
            "structure_1": f"{m1.spacegroup_symbol} ({m1.spacegroup_number})",
            "structure_2": f"{m2.spacegroup_symbol} ({m2.spacegroup_number})",
        },
        "packing_fraction_delta_percent": relative_delta(
            m1.packing_fraction, m2.packing_fraction
        ),
        "avg_electronegativity_delta": m2.avg_electronegativity - m1.avg_electronegativity,
        "avg_atomic_mass_delta": m2.avg_atomic_mass - m1.avg_atomic_mass,
        "avg_valence_electrons_delta": m2.avg_valence_electrons - m1.avg_valence_electrons,
    }

    return {
        "structure_1": asdict(m1),
        "structure_2": asdict(m2),
        "geometry_comparison": geometry_comparison,
        "physical_comparison": physical_comparison,
    }


def print_human_readable(result: dict[str, Any], cif1: Path, cif2: Path) -> None:
    s1 = result["structure_1"]
    s2 = result["structure_2"]
    geo = result["geometry_comparison"]
    phy = result["physical_comparison"]

    print("=" * 80)
    print("CIF 晶体结构对比报告")
    print("=" * 80)
    print(f"结构 1: {cif1}")
    print(f"结构 2: {cif2}")
    print()

    print("[基础信息]")
    print(
        f"- 结构1: {s1['formula']} | 空间群 {s1['spacegroup_symbol']} ({s1['spacegroup_number']}) | "
        f"密度 {s1['density']:.4f} g/cm^3 | 体积 {s1['volume']:.4f} Å^3"
    )
    print(
        f"- 结构2: {s2['formula']} | 空间群 {s2['spacegroup_symbol']} ({s2['spacegroup_number']}) | "
        f"密度 {s2['density']:.4f} g/cm^3 | 体积 {s2['volume']:.4f} Å^3"
    )
    print()

    print("[几何层面对比]")
    print(f"- 晶胞体积变化: {geo['volume_delta_percent']:.3f}%")
    print(f"- 密度变化: {geo['density_delta_percent']:.3f}%")
    print(
        "- 晶格常数变化(%): "
        f"a={geo['lattice_delta_percent']['a']:.3f}, "
        f"b={geo['lattice_delta_percent']['b']:.3f}, "
        f"c={geo['lattice_delta_percent']['c']:.3f}"
    )
    print(
        "- 晶格角变化(°): "
        f"α={geo['angle_delta_deg']['alpha']:.3f}, "
        f"β={geo['angle_delta_deg']['beta']:.3f}, "
        f"γ={geo['angle_delta_deg']['gamma']:.3f}"
    )
    print(f"- 最近邻平均距离变化: {geo['avg_nearest_neighbor_delta_percent']:.3f}%")
    print(f"- 3Å 内平均配位数变化: {geo['coordination_delta_percent']:.3f}%")
    print(f"- RDF 余弦相似度: {geo['rdf_cosine_similarity']:.4f} (越接近 1 越相似)")
    print(f"- StructureMatcher 同构判定: {geo['structure_matcher_fit']}")
    print(f"- StructureMatcher RMS 位移: {geo['structure_matcher_rms']}")
    print()

    print("[物理层面对比]")
    print(f"- 化学式是否一致: {phy['formula_equal']}")
    print(f"- 空间群是否一致: {phy['spacegroup_equal']}")
    print(
        f"- 空间群变化: {phy['spacegroup_change']['structure_1']} -> "
        f"{phy['spacegroup_change']['structure_2']}"
    )
    print(f"- 填充因子变化: {phy['packing_fraction_delta_percent']:.3f}%")
    print(f"- 平均电负性变化: {phy['avg_electronegativity_delta']:.4f}")
    print(f"- 平均原子质量变化: {phy['avg_atomic_mass_delta']:.4f}")
    print(f"- 平均价电子数变化: {phy['avg_valence_electrons_delta']:.4f}")


def main() -> None:
    args = parse_args()
    s1 = Structure.from_file(args.cif1)
    s2 = Structure.from_file(args.cif2)

    s1 = preprocess_structure(s1, standardize=args.standardize, primitive=args.primitive)
    s2 = preprocess_structure(s2, standardize=args.standardize, primitive=args.primitive)

    result = compare_structures(s1, s2, rdf_cutoff=args.rdf_cutoff, rdf_bin=args.rdf_bin)
    print_human_readable(result, args.cif1, args.cif2)

    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 结果已写入: {args.json_out}")


if __name__ == "__main__":
    main()
