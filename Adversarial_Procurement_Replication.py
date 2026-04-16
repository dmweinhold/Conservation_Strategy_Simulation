
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conservation Strategy Game — Replication-Ready Simulator
=========================================================

Author: Diana Weinhold
Contact: d.weinhold@lse.ac.uk
All rights reserved.

If you use this simulator in teaching, research, or publications, please cite:

Weinhold, D. & Andersen, L. E. (2026).
"Adversarial Procurement in Two-Value Space: Insights and Evidence for Conservation Siting"
London School of Economics Working Paper.

DOI: 10.5281/zenodo.17114490
GitHub: https://github.com/dmweinhold/
"""


"""
Adversarial Procurement — Replication-Ready Simulator (v2)
=============================================================

This single script reproduces all analyses:
- Static final outcomes (vs leakage)
- Dynamic trajectories (time paths)
- PSE/DLE Effects table (decomposition)

Includes:
- Event-based Additionality (strict) with end-of-game residual bump,
  and a final post-residual datapoint in trajectories.
- Optional per-round batching (multi-claims per round).
- Strategic Farmer 'risky/safe' selection logic (configurable).
- Strategy aliases (max_diff|max_difference).


REQUIREMENTS
------------
pip install numpy pandas matplotlib pyyaml


OUTPUTS
-------
Figures (*.png) and tables (*.csv) saved into ./replication_outputs/ by default.
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# User Interface
# --------------------------

def launch_ui_and_get_args():
    import tkinter as tk
    from tkinter import ttk

    vals = {
        'world': 'claims',
        'reps': 500,
        'rho': 0.0,
        'alloc': 'equal',
        'farmer_pct': 0.7,
        'claims_greens': ['max_env', 'hot_spot', 'block_farmers', 'max_diff', 'random'],
        'budget_rules': ['maxE_budget', 'hotspot_budget', 'block_budget', 'max_efficiency_budget'],
        'farmers': ['naive', 'strategic'],
        'leakages': '1.0,0.75,0.5,0.25,0.0',
        'grid': 10,
        'rounds': 'auto',
        'outdir': 'replication_outputs',
        'seed': 42,
    }

    root = tk.Tk()
    root.title('Conservation Strategy Game — Replication Package')

    PADX, PADY = 8, 6
    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky='nsew')

    ttk.Label(frm, text='World').grid(row=0, column=0, sticky='w', padx=PADX, pady=PADY)
    world_var = tk.StringVar(value=vals['world'])
    ttk.Combobox(frm, values=['claims', 'budget'], textvariable=world_var,
                 width=12, state='readonly').grid(row=0, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Replications').grid(row=1, column=0, sticky='w', padx=PADX, pady=PADY)
    reps_var = tk.IntVar(value=vals['reps'])
    ttk.Spinbox(frm, from_=10, to=100000, increment=10, textvariable=reps_var,
                width=10).grid(row=1, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Correlation (ρ)').grid(row=2, column=0, sticky='w', padx=PADX, pady=PADY)
    rho_var = tk.DoubleVar(value=vals['rho'])
    tk.Scale(frm, from_=-1.0, to=1.0, resolution=0.1, orient='horizontal',
             variable=rho_var, length=240).grid(row=2, column=1, columnspan=3,
                                                sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Allocation').grid(row=3, column=0, sticky='w', padx=PADX, pady=PADY)
    alloc_var = tk.StringVar(value=vals['alloc'])
    alloc_cb = ttk.Combobox(frm, values=['equal', 'political'], textvariable=alloc_var,
                            width=12, state='readonly')
    alloc_cb.grid(row=3, column=1, sticky='w', padx=PADX, pady=PADY)

    farmer_pct_row = ttk.Frame(frm)
    farmer_pct_row.grid(row=4, column=0, columnspan=4, sticky='w', padx=PADX, pady=PADY)
    ttk.Label(farmer_pct_row, text='Farmer % (if political)').grid(row=0, column=0, sticky='w')
    farmer_pct_var = tk.DoubleVar(value=vals['farmer_pct'])
    tk.Scale(farmer_pct_row, from_=0.0, to=1.0, resolution=0.05, orient='horizontal',
             variable=farmer_pct_var, length=240).grid(row=0, column=1, sticky='w', padx=(PADX, 0))

    def _toggle_farmer_pct(*_):
        if alloc_var.get() == 'political':
            farmer_pct_row.grid()
        else:
            farmer_pct_row.grid_remove()

    alloc_cb.bind('<<ComboboxSelected>>', _toggle_farmer_pct)
    _toggle_farmer_pct()

    ttk.Label(frm, text='Farmer strategies').grid(row=5, column=0, sticky='w', padx=PADX, pady=PADY)
    farmers_frame = ttk.Frame(frm)
    farmers_frame.grid(row=5, column=1, columnspan=4, sticky='w', padx=PADX, pady=PADY)
    farmer_vars = {}
    for i, f in enumerate(['naive', 'strategic']):
        v = tk.BooleanVar(value=(f in vals['farmers']))
        farmer_vars[f] = v
        ttk.Checkbutton(farmers_frame, text=f, variable=v).grid(row=0, column=i, sticky='w', padx=(8, 16))

    claims_frame = ttk.Labelframe(frm, text='Claims World green strategies', padding=8)
    claims_frame.grid(row=6, column=0, columnspan=4, sticky='we', padx=PADX, pady=PADY)
    claims_vars = {}
    claims_list = ['max_env', 'hot_spot', 'block_farmers', 'max_diff', 'random']
    for i, g in enumerate(claims_list):
        v = tk.BooleanVar(value=(g in vals['claims_greens']))
        claims_vars[g] = v
        ttk.Checkbutton(claims_frame, text=g, variable=v).grid(
            row=i // 3, column=i % 3, sticky='w', padx=(8, 16), pady=2
        )

    budget_frame = ttk.Labelframe(frm, text='Budget World green rules', padding=8)
    budget_frame.grid(row=7, column=0, columnspan=4, sticky='we', padx=PADX, pady=PADY)
    budget_vars = {}
    budget_list = ['maxE_budget', 'hotspot_budget', 'block_budget', 'max_efficiency_budget']
    for i, r in enumerate(budget_list):
        v = tk.BooleanVar(value=(r in vals['budget_rules']))
        budget_vars[r] = v
        ttk.Checkbutton(budget_frame, text=r, variable=v).grid(
            row=i // 2, column=i % 2, sticky='w', padx=(8, 16), pady=2
        )

    def _toggle_world_panels(*_):
        if world_var.get() == 'claims':
            claims_frame.grid()
            budget_frame.grid_remove()
        else:
            budget_frame.grid()
            claims_frame.grid_remove()

    world_var.trace_add('write', lambda *_: _toggle_world_panels())
    _toggle_world_panels()

    ttk.Label(frm, text='Leakages (comma)').grid(row=8, column=0, sticky='w', padx=PADX, pady=PADY)
    leak_var = tk.StringVar(value=vals['leakages'])
    ttk.Entry(frm, textvariable=leak_var, width=20).grid(row=8, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Grid size (n)').grid(row=9, column=0, sticky='w', padx=PADX, pady=PADY)
    grid_var = tk.IntVar(value=vals['grid'])
    ttk.Spinbox(frm, from_=4, to=30, increment=1, textvariable=grid_var,
                width=8).grid(row=9, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Rounds ("auto" or int)').grid(row=10, column=0, sticky='w', padx=PADX, pady=PADY)
    rounds_var = tk.StringVar(value=str(vals['rounds']))
    ttk.Entry(frm, textvariable=rounds_var, width=12).grid(row=10, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Output folder').grid(row=11, column=0, sticky='w', padx=PADX, pady=PADY)
    outdir_var = tk.StringVar(value=vals['outdir'])
    ttk.Entry(frm, textvariable=outdir_var, width=24).grid(row=11, column=1, sticky='w', padx=PADX, pady=PADY)

    ttk.Label(frm, text='Seed').grid(row=12, column=0, sticky='w', padx=PADX, pady=PADY)
    seed_var = tk.IntVar(value=vals['seed'])
    ttk.Spinbox(frm, from_=0, to=10_000_000, increment=1, textvariable=seed_var,
                width=10).grid(row=12, column=1, sticky='w', padx=PADX, pady=PADY)

    chosen = {}

    def run_and_close():
        sel_claims = [g for g, v in claims_vars.items() if v.get()]
        if not sel_claims:
            sel_claims = ['max_env', 'hot_spot', 'block_farmers']

        sel_budget = [r for r, v in budget_vars.items() if v.get()]
        if not sel_budget:
            sel_budget = ['maxE_budget', 'hotspot_budget', 'block_budget', 'max_efficiency_budget']

        sel_farmers = [f for f, v in farmer_vars.items() if v.get()]
        if not sel_farmers:
            sel_farmers = ['naive', 'strategic']

        chosen['world'] = world_var.get()
        chosen['reps'] = int(reps_var.get())
        chosen['rho'] = float(rho_var.get())
        chosen['alloc'] = alloc_var.get()
        chosen['farmer_pct'] = float(farmer_pct_var.get())
        chosen['claims_greens'] = ','.join(sel_claims)
        chosen['budget_rules'] = ','.join(sel_budget)
        chosen['farmer'] = sel_farmers
        chosen['leakages'] = leak_var.get().strip()
        chosen['grid'] = int(grid_var.get())
        chosen['rounds'] = rounds_var.get().strip()
        chosen['outdir'] = outdir_var.get().strip()
        chosen['seed'] = int(seed_var.get())
        root.destroy()

    ttk.Button(frm, text='Run', command=run_and_close).grid(
        row=13, column=0, pady=(PADY + 2, PADY), padx=PADX, sticky='w'
    )
    ttk.Button(frm, text='Cancel', command=root.destroy).grid(
        row=13, column=1, pady=(PADY + 2, PADY), padx=PADX, sticky='w'
    )

    root.mainloop()
    return chosen or None



# -------- UI default toggle --------
# Set to False to disable the popup menu and use CLI args only.
USE_UI_DEFAULT = True

def _gui_available() -> bool:
    """Return True if Tkinter can start (non-headless), False otherwise."""
    try:
        import tkinter as _tk
        root = _tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        return True
    except Exception:
        return False

# --------------------------
# Utils / I-O
# --------------------------

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def export_claims_effects_table(
    outdir: Path,
    greens_selected: List[str],
    farmer_strategy: str,
    leakages: List[float],
    *,
    rho: float, grid: int, rounds: int,
    claims_green: int, claims_farmer: int,
    reps: int, seed: Optional[int],
    multi_per_round: bool, risky_rule: str,
    filename: Optional[str] = None
) -> Path:
    """
    Writes a CSV summarizing Claims-World effects across (strategy, leakage).
    If your code defines compute_effects_table(...), we use it. Otherwise, we fall back to
    a lightweight builder using run_replications(...).
    """
    outdir = Path(outdir)
    fname = filename or f'claims_effects_table_{farmer_strategy}.csv'
    path = outdir / fname

    # Preferred: use the project’s canonical table maker if present
    if 'compute_effects_table' in globals():
        df = compute_effects_table(
            greens=greens_selected,
            farmer_strategy=farmer_strategy,
            leakages=leakages,
            rho=rho, grid=grid, rounds=rounds,
            claims_green=claims_green, claims_farmer=claims_farmer,
            reps=reps, seed=seed,
            multi_per_round=multi_per_round, risky_rule=risky_rule
        )
        df.to_csv(path, index=False)
        print(f"[saved] {path}")
        return path

    # Fallback: build a concise summary from run_replications(...)
    rows = []
    for g in greens_selected:
        for L in leakages:
            res = run_replications(
                green_strategy=g, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed,
                multi_per_round=multi_per_round, risky_rule=risky_rule
            )
            rows.append({
                'farmer_strategy': farmer_strategy,
                'green_strategy' : g,
                'leakage_pct'    : int(round(100*L)),
                # Means
                'mean_conservation': round(float(res.get('mean_cons', np.nan)), 3),
                'mean_additionality': round(float(res.get('mean_addl', np.nan)), 3),
                'mean_welfare_gap' : round(float(res.get('mean_wgap', np.nan)), 3),
                # If your engine returns these, include; else they'll be NaN
                'mean_pse' : round(float(res.get('mean_pse', np.nan)), 3) if res.get('mean_pse') is not None else np.nan,
                'mean_dle' : round(float(res.get('mean_dle', np.nan)), 3) if res.get('mean_dle') is not None else np.nan,
            })
    df = pd.DataFrame(rows).sort_values(['green_strategy','leakage_pct'])
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


def export_budget_effects_summary(
    outdir: Path,
    farmer_strategy: str,
    rules: List[str],
    leakages: List[float],
    *,
    rho: float, grid: int, rounds: int,
    B_G: float | None,
    reps: int, seed: Optional[int],
    theta: float,
    hotspot_additive: bool,
    multi_per_round: bool,
    farmer_share: float | None = 0.5,
    filename: Optional[str] = None
) -> Path:
    """
    Writes a CSV summarizing Budget-World effects across (rule, leakage).
    Includes purchased-only and final (incl. DLE where credited by engine).
    """
    outdir = Path(outdir)
    fname = filename or f'budget_effects_summary_{farmer_strategy}.csv'
    path = outdir / fname

    rows, cov = [], []
    for rule in rules:
        for L in leakages:
            res = run_replications_budget(
                green_rule=rule, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds, B_G=B_G,
                reps=reps, seed=seed, theta=theta,
                hotspot_additive=hotspot_additive,
                multi_per_round=multi_per_round,
                farmer_share=farmer_share
            )
            cov.append(res.get('coverage_rate', 1.0))
            rows.append({
                'farmer_strategy': farmer_strategy,
                'green_rule': rule,
                'leakage_pct': int(round(100 * L)),
                'mean_pse': round(float(res.get('mean_pse', np.nan)), 3),
                'mean_addl_pure': round(float(res.get('mean_addl_pure', np.nan)), 3) if res.get('mean_addl_pure') is not None else np.nan,
                'mean_wgap_pure': round(float(res.get('mean_wgap_pure', np.nan)), 3) if res.get('mean_wgap_pure') is not None else np.nan,
                'mean_conservation': round(float(res.get('mean_cons', np.nan)), 3),
                'mean_additionality': round(float(res.get('mean_addl', np.nan)), 3),
                'mean_welfare_gap': round(float(res.get('mean_wgap', np.nan)), 3),
                'allocation': 'political' if farmer_share not in (None, 0.5) else 'equal',
                'farmer_share': farmer_share,
                'B_G': B_G,
            })
    df = pd.DataFrame(rows).sort_values(['green_rule', 'leakage_pct'])
    df.to_csv(path, index=False)
    print(f'[saved] {path} (coverage~{np.mean(cov):.3f})')
    return path

# --------------------------
# Grid generation (Gaussian copula for corr(e,a))
# --------------------------

def initialize_grid(n: int = 10,
                    rho: float = 0.0,
                    low: float = 0.1,
                    high: float = 10.0,
                    ag_unsuitable_prob: float = 0.0,
                    jitter_ties: bool = True,
                    rng: Optional[np.random.Generator] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Gaussian-copula initialization of (E, A) with optional features
    mirroring the legacy function:
      - ag_unsuitable_prob: with this probability, set A to a random negative value,
        simulating agronomically unsuitable plots (legacy: U(-10,0)).
      - jitter_ties: if any exact E==A after rescaling, add a tiny jitter to E
        to break tie ordering (legacy nudged by +/- 1.0).

    Returns:
        E, A: flattened arrays of length n*n in [low, high] (A may be <0 for unsuitable)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw correlated standard normals
    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n * n)

    # Vectorized standard normal CDF via robust erf
    try:
        from scipy.special import erf as _erf
        u_e = 0.5 * (1.0 + _erf(z[:, 0] / np.sqrt(2.0)))
        u_a = 0.5 * (1.0 + _erf(z[:, 1] / np.sqrt(2.0)))
    except Exception:
        # Abramowitz–Stegun 7.1.26 vectorized approximation (max abs err ~1.5e-7)
        a1,a2,a3,a4,a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        def _erf_vec(x):
            sign = np.sign(x); ax = np.abs(x)
            t = 1.0 / (1.0 + p * ax)
            y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
            return sign * (1.0 - y * np.exp(-(ax * ax)))
        u_e = 0.5 * (1.0 + _erf_vec(z[:, 0] / np.sqrt(2.0)))
        u_a = 0.5 * (1.0 + _erf_vec(z[:, 1] / np.sqrt(2.0)))

    # Rescale to [low, high]
    E = low + (high - low) * u_e
    A = low + (high - low) * u_a

    # Optional: agronomically unsuitable probability -> negative A
    if ag_unsuitable_prob and ag_unsuitable_prob > 0.0:
        mask = rng.random(E.size) < ag_unsuitable_prob
        if np.any(mask):
            A[mask] = rng.uniform(-10.0, 0.0, size=mask.sum())

    # Optional: break any exact ties E==A (extremely rare; legacy nudged by ±1.0)
    if jitter_ties:
        tie_mask = (E == A)
        if np.any(tie_mask):
            # small epsilon consistent with value range; avoids changing rankings materially
            eps = (high - low) * 1e-6
            E[tie_mask] += rng.choice([-eps, eps], size=tie_mask.sum())

    return E, A




# --------------------------
# Claims allocation
# --------------------------

def allocate_points(total_cells: int,
                    allocation: str = 'equal',
                    farmer_percentage: float = 0.7
                    ) -> Tuple[int, int]:
    """
    Determine claims for Greens and Farmers.
    'equal'      => 50/50 split (rounded to sum to total_cells)
    'political'  => farmer_percentage to Farmers, remainder to Greens.
    """
    if allocation == 'equal':
        cf = total_cells // 2
        cg = total_cells - cf
        return cg, cf
    elif allocation == 'political':
        cf = int(round(total_cells * farmer_percentage))
        cf = max(0, min(total_cells, cf))
        cg = total_cells - cf
        return cg, cf
    else:
        raise ValueError("allocation must be 'equal' or 'political'")


# --------------------------
# Farmer BAU construction
# --------------------------

def farmer_bau_set(E: np.ndarray,
                   A: np.ndarray,
                   claims_farmer: int,
                   claims_green: int,
                   farmer_strategy: str = 'naive',
                   risky_rule: str = 'green_claims'  # or 'farmer_claims'
                   ) -> List[int]:
    """
    Construct the Farmer BAU set (indices) assuming no interference by Greens.
    - naive: top A
    - strategic: prioritize 'risky' plots that Greens likely want (by E),
                 then fill by A from 'safe'.
      risky_rule:
        - 'green_claims'  : risky count = claims_green (default)
        - 'farmer_claims' : risky count = claims_farmer
    """
    n = E.size
    idx_all = np.arange(n)

    if farmer_strategy == 'naive':
        order = np.argsort(-A)
        return order[:claims_farmer].tolist()

    elif farmer_strategy == 'strategic':
        risky_count = claims_green if risky_rule == 'green_claims' else claims_farmer
        risky_count = max(0, min(n, risky_count))
        risky = np.argsort(-E)[:risky_count]
        risky_mask = np.zeros(n, dtype=bool)
        risky_mask[risky] = True

        risky_by_A = risky[np.argsort(-A[risky])]
        safe = idx_all[~risky_mask]
        safe_by_A = safe[np.argsort(-A[safe])]

        chosen = []
        for i in range(risky_by_A.size):
            if len(chosen) < claims_farmer:
                chosen.append(int(risky_by_A[i]))
            else:
                break
        if len(chosen) < claims_farmer:
            need = claims_farmer - len(chosen)
            chosen += [int(x) for x in safe_by_A[:need]]
        return chosen

    else:
        raise ValueError("farmer_strategy must be 'naive' or 'strategic'")


# --------------------------
# Green move rules (+ aliases)
# --------------------------

def normalize_green_strategy_name(s: str) -> str:
    s = s.strip().lower()
    if s == 'max_difference':
        return 'max_diff'
    return s

def select_green_move(strategy: str,
                      E: np.ndarray,
                      A: np.ndarray,
                      unclaimed: np.ndarray,
                      rng: np.random.Generator,
                      alpha: float = 1.0,
                      beta: float = 1.0
                      ) -> Optional[int]:
    """Choose one index from unclaimed according to the strategy."""
    if unclaimed.size == 0:
        return None
    s = normalize_green_strategy_name(strategy)
    if s == 'max_env':
        i = unclaimed[np.argmax(E[unclaimed])]
        return int(i)
    elif s == 'block_farmers':
        i = unclaimed[np.argmax(A[unclaimed])]
        return int(i)
    elif s == 'hot_spot':
        score = (E[unclaimed] ** alpha) * (A[unclaimed] ** beta)
        i = unclaimed[np.argmax(score)]
        return int(i)
    elif s == 'max_diff':
        score = E[unclaimed] - A[unclaimed]
        i = unclaimed[np.argmax(score)]
        return int(i)
    elif s == 'random':
        i = int(rng.choice(unclaimed))
        return i
    else:
        raise ValueError(f"Unknown Green strategy: {strategy}")


def select_farmer_move(farmer_strategy: str,
                       E: np.ndarray,
                       A: np.ndarray,
                       unclaimed: np.ndarray,
                       risky_mask: Optional[np.ndarray] = None
                       ) -> Optional[int]:
    """
    Farmer move:
    - naive: pick highest A among unclaimed
    - strategic: pick highest A among 'risky' set first; else among safe.
    """
    if unclaimed.size == 0:
        return None
    if farmer_strategy == 'naive' or risky_mask is None:
        i = unclaimed[np.argmax(A[unclaimed])]
        return int(i)
    else:
        risky_unclaimed = unclaimed[risky_mask[unclaimed]]
        if risky_unclaimed.size > 0:
            i = risky_unclaimed[np.argmax(A[risky_unclaimed])]
            return int(i)
        safe_unclaimed = unclaimed[~risky_mask[unclaimed]]
        if safe_unclaimed.size > 0:
            i = safe_unclaimed[np.argmax(A[safe_unclaimed])]
            return int(i)
        return None


# --------------------------
# Green BAU indices + scalar benchmark
# --------------------------

def greens_bau_indices(E: np.ndarray,
                       farmer_bau: List[int],
                       claims_green: int) -> List[int]:
    """Top-E indices for Greens after removing Farmer BAU, size == claims_green."""
    n = E.size
    all_idx = np.arange(n)
    rem = np.setdiff1d(all_idx, np.array(farmer_bau), assume_unique=False)
    if rem.size == 0 or claims_green <= 0:
        return []
    pick = rem[np.argsort(-E[rem])[:min(claims_green, rem.size)]]
    return pick.tolist()


# --------------------------
# Welfare metric
# --------------------------

def optimal_total(E: np.ndarray, A: np.ndarray) -> float:
    return float(np.maximum(E, A).sum())


# --------------------------
# Engine: simulate one game with event-based additionality
# --------------------------

def simulate_game_dynamic_BAU(E: np.ndarray,
                              A: np.ndarray,
                              green_strategy: str,
                              farmer_strategy: str,
                              leakage: float,
                              claims_green: int,
                              claims_farmer: int,
                              rounds: int = 50,
                              rng: Optional[np.random.Generator] = None,
                              alpha: float = 1.0,
                              beta: float = 1.0,
                              multi_per_round: bool = False,
                              risky_rule: str = 'green_claims'  # or 'farmer_claims'
                              ) -> Dict[str, object]:
    """
    Simulate the sequential game with event-based additionality.
    Farmer moves first each round, then Green.
    Leakage rule: deductions = floor(Green_BAU_claims * (1 - leakage)).
    End: residual unclaimed plots go to Greens (DLE); any residual Farmer-BAU plots
         also produce a +E jump to additionality.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = E.size
    all_idx = np.arange(n)

    # Optional invariant guard (comment out if you relax this later)
    # assert claims_green + claims_farmer == n, "BAU complement equivalence assumes Cg+Cf=N."

    # --- BAU sets ---
    farmer_bau = farmer_bau_set(E, A, claims_farmer, claims_green,
                                farmer_strategy, risky_rule=risky_rule)
    farmer_bau_mask = np.zeros(n, dtype=bool); farmer_bau_mask[farmer_bau] = True

    green_bau_idx = greens_bau_indices(E, farmer_bau, claims_green)
    green_bau_mask = np.zeros(n, dtype=bool); green_bau_mask[green_bau_idx] = True

    greens_bau_score = float(E[green_bau_idx].sum())
    opt_total = optimal_total(E, A)

    # Risky mask for strategic farmer (aligned with BAU risky rule)
    risky_mask = None
    if farmer_strategy == 'strategic':
        risky_count = claims_green if risky_rule == 'green_claims' else claims_farmer
        risky_count = max(0, min(n, risky_count))
        risky = np.argsort(-E)[:risky_count]
        risky_mask = np.zeros(n, dtype=bool)
        risky_mask[risky] = True

    # --- State ---
    unclaimed = all_idx.copy()
    Sg: List[int] = []
    Sf: List[int] = []
    cg_rem = claims_green
    cf_rem = claims_farmer

    green_BAU_claims = 0
    farmer_deductions_applied = 0

    # Event-based additionality
    addl_event = 0.0

    # Per-round series (PSE; Conservation pre-residual; Additionality event-sum; Welfare gap)
    rounds_list: List[int] = []
    pse_series: List[float] = []
    cons_series: List[float] = []
    addl_series: List[float] = []
    welfare_gap_series: List[float] = []

    def record_round(rnum: int):
        pse = float(E[Sg].sum()) if Sg else 0.0
        cons_now = pse  # pre-residual (PSE only)
        addl_now = addl_event  # event-based trajectory
        welfare_actual = float(A[Sf].sum() + E[Sg].sum())
        welfare_gap = 100.0 * (opt_total - welfare_actual) / opt_total

        rounds_list.append(rnum)
        pse_series.append(pse)
        cons_series.append(cons_now)
        addl_series.append(addl_now)
        welfare_gap_series.append(welfare_gap)

    # --- Gameplay ---
    for r in range(1, rounds + 1):
        if unclaimed.size == 0 or (cg_rem <= 0 and cf_rem <= 0):
            break

        # Points to spend this round
        if multi_per_round:
            rounds_left = rounds - r + 1
            farmer_points_to_use = min(cf_rem, max(1, math.ceil(cf_rem / rounds_left))) if cf_rem > 0 else 0
            green_points_to_use  = min(cg_rem, max(1, math.ceil(cg_rem / rounds_left))) if cg_rem > 0 else 0
        else:
            farmer_points_to_use = 1 if cf_rem > 0 else 0
            green_points_to_use  = 1 if cg_rem > 0 else 0

        # FARMER moves (up to farmer_points_to_use)
        for _ in range(farmer_points_to_use):
            if cf_rem <= 0 or unclaimed.size == 0:
                break
            f_pick = select_farmer_move(farmer_strategy, E, A, unclaimed, risky_mask)
            if f_pick is None:
                break
            Sf.append(f_pick)
            # Event-based additionality: Farmer takes a Green-BAU plot => -E
            if green_bau_mask[f_pick]:
                addl_event -= float(E[f_pick])

            # remove and decrement
            unclaimed = unclaimed[unclaimed != f_pick]
            cf_rem -= 1

        # GREEN moves (up to green_points_to_use)
        for _ in range(green_points_to_use):
            if cg_rem <= 0 or unclaimed.size == 0:
                break
            g_pick = select_green_move(green_strategy, E, A, unclaimed, rng, alpha, beta)
            if g_pick is None:
                break
            Sg.append(g_pick)
            # Event-based additionality: Green takes a Farmer-BAU plot => +E
            if farmer_bau_mask[g_pick]:
                addl_event += float(E[g_pick])
                # Apply leakage deductions based on cumulative Green BAU hits
                green_BAU_claims += 1
                expected_deductions = int(math.floor(green_BAU_claims * (1.0 - leakage)))
                to_apply = expected_deductions - farmer_deductions_applied
                if to_apply > 0:
                    cf_rem = max(0, cf_rem - to_apply)
                    farmer_deductions_applied += to_apply

            # remove and decrement
            unclaimed = unclaimed[unclaimed != g_pick]
            cg_rem -= 1

        # Record per-round series (pre-residual)
        record_round(r)

        if unclaimed.size == 0 or (cg_rem <= 0 and cf_rem <= 0):
            break

    # --- End-game residuals to Greens (DLE) ---
    pre_residual_Sg = set(Sg)
    if unclaimed.size > 0:
        # assign residuals to Greens
        Sg.extend(unclaimed.tolist())

        # residual additionality event: +E for residual Farmer-BAU cells
        residual = np.array(list(set(Sg) - pre_residual_Sg), dtype=int)
        if residual.size > 0:
            res_bau = residual[farmer_bau_mask[residual]]
            if res_bau.size > 0:
                addl_event += float(E[res_bau].sum())

        unclaimed = np.array([], dtype=int)

    # Final aggregates
    pse_final = float(pse_series[-1]) if pse_series else 0.0
    cons_final = float(E[Sg].sum())
    dle_final  = cons_final - pse_final

    addl_final_event = float(addl_event)
    addl_final_bench = cons_final - greens_bau_score
    if abs(addl_final_event - addl_final_bench) > 1e-6:
        print(f"[warn] addl mismatch: event={addl_final_event:.6f} vs bench={addl_final_bench:.6f}")

    welfare_actual_final = float(A[Sf].sum() + E[Sg].sum())
    welfare_gap_final    = 100.0 * (opt_total - welfare_actual_final) / opt_total

    # Append a final (post-residual) datapoint
    rounds_list.append((rounds_list[-1] + 1) if rounds_list else 1)
    pse_series.append(pse_final)               # unchanged by residual
    cons_series.append(cons_final)             # includes DLE
    addl_series.append(addl_final_event)       # includes residual bump (if any)
    welfare_gap_series.append(welfare_gap_final)

    return dict(
        Sg=Sg, Sf=Sf, farmer_bau=farmer_bau, green_bau=green_bau_idx,
        rounds=rounds_list,
        pse=pse_series, conservation=cons_series, additionality=addl_series, welfare_gap=welfare_gap_series,
        pse_final=pse_final, dle_final=dle_final, cons_final=cons_final,
        addl_final=addl_final_event,  # equals BAU-difference up to FP tolerance
        welfare_gap_final=welfare_gap_final
    )


# =========================
# Budgeted contest extension
# =========================

def calibrate_budgets(
    A: np.ndarray,
    claims_farmer: int | None = None,
    farmer_share: float | None = 0.5,
    B_G_override: float | None = None
) -> tuple[float, float, float, bool]:
    """
    Calibrate (B_F_init, B_G_init, A_tot, is_coverage) for budget mode.

    Priority order:
    - If farmer_share is not None: coverage baseline per grid:
        B_F = farmer_share * sum(A); B_G = (1 - farmer_share) * sum(A)
        -> coverage (B_F+B_G == A_tot)
    - Else if claims_farmer is not None: 'claims-consistent farmer' baseline:
        B_F = sum of top 'claims_farmer' prices; B_G = A_tot - B_F
        -> coverage (B_F+B_G == A_tot)
    - Else if B_G_override is not None: set B_G; set B_F = A_tot - B_G
        -> coverage (still equals A_tot)
    - Else fallback: equal coverage shares (0.5/0.5).

    You can still pass B_G_override together with farmer_share if you
    explicitly want under/over-funding; in that case we mark is_coverage=False.
    """
    A_sorted = np.sort(A)[::-1]
    A_tot = float(A_sorted.sum())

    # default: coverage
    if farmer_share is not None:
        B_F = float(farmer_share * A_tot)
        B_G = float((1.0 - farmer_share) * A_tot)
        is_coverage = True
    elif claims_farmer is not None and claims_farmer > 0:
        B_F = float(A_sorted[:claims_farmer].sum())
        B_G = float(A_tot - B_F)
        is_coverage = True
    else:
        # fallback coverage 50/50
        B_F = 0.5 * A_tot
        B_G = 0.5 * A_tot
        is_coverage = True

    # if an explicit B_G_override is provided, honor it (may break coverage)
    if B_G_override is not None:
        B_G = float(B_G_override)
        B_F = float(A_tot - B_G)
        # recompute coverage flag (tolerant to FP error)
        is_coverage = abs((B_F + B_G) - A_tot) <= 1e-6

    return B_F, B_G, A_tot, is_coverage

def farmer_bau_budget_indices(A: np.ndarray,
                              B_F: float) -> List[int]:
    """
    Farmer BAU purchase set under budgets: take items by descending A until B_F is exhausted.
    Returns list of indices in purchase order.
    """
    order = np.argsort(-A)  # high A first
    picked = []
    cost = 0.0
    for i in order:
        if cost + A[i] <= B_F:
            picked.append(int(i))
            cost += float(A[i])
    return picked


def green_bau_budget_indices(E: np.ndarray,
                             A: np.ndarray,
                             remaining_idx: np.ndarray,
                             B_G: float) -> List[int]:
    """
    Strategy-invariant BAU for Greens under budgets = Max-E subject to budget on the
    Farmer-BAU complement (mirrors paper's 'strategy-invariant' BAU idea).
    Greedy by E with affordability.
    """
    # Sort remaining by E descending, buy while affordable
    rem = np.array(remaining_idx, dtype=int)
    order = rem[np.argsort(-E[rem])]
    picked, spend = [], 0.0
    for i in order:
        ai = float(A[i])
        if spend + ai <= B_G:
            picked.append(int(i))
            spend += ai
    return picked


def compute_lambda_for_budget(E: np.ndarray,
                              A: np.ndarray,
                              remaining_idx: np.ndarray,
                              B_remain: float,
                              theta: float = 0.0,
                              max_iter: int = 40,
                              tol: float = 1e-6) -> float:
    """
    Find lambda_t so that picking {i: E_i - (lambda + theta) A_i > 0} among remaining
    would (approximately) spend the Green remaining budget. This is a simple bisection
    using the 'positive-score set spends ~ B_remain' heuristic. Items with A_i > B_remain
    cannot be bought in one move but we still use them to calibrate lambda as a 'cutoff'.
    """
    if B_remain <= tol or remaining_idx.size == 0:
        return 0.0

    # Safe upper bound based on E/A ratios (subtract theta)
    # --- robust high bracket ---
    ratios = E[remaining_idx] / np.maximum(A[remaining_idx], 1e-12)
    _eps = 1e-9  # tiny buffer to avoid sitting exactly on the frontier
    lam_hi = float(np.max(ratios) - theta - _eps)
    lam_lo = 0.0
    lam_hi = max(lam_hi, 0.0)


    def spend_at(lam: float) -> float:
        score = E[remaining_idx] - (lam + theta) * A[remaining_idx]
        sel = remaining_idx[score >= -_eps]
        # Approximate 'would spend' by summing costs of the positive-score set
        return float(np.sum(A[sel])) if sel.size > 0 else 0.0

    s_lo, s_hi = spend_at(lam_lo), spend_at(lam_hi)
    # If even lam_hi yields spend <= B, accept lam_hi (very selective frontier)
    if s_hi <= B_remain:
        return lam_hi
    # If lam=0 already spends less than budget, no need to penalize costs further
    if s_lo <= B_remain:
        return lam_lo

    lam = 0.0
    for _ in range(max_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        s_mid = spend_at(lam)
        if abs(s_mid - B_remain) <= tol:
            break
        if s_mid > B_remain:
            lam_lo = lam  # need to penalize cost more
        else:
            lam_hi = lam
    return lam

def green_pick_maxE_pure_budget(E: np.ndarray,
                                A: np.ndarray,
                                unclaimed: np.ndarray,
                                B_G_remain: float) -> Optional[int]:
    """
    Pure 'Max Environment' under a budget: pick the highest-E affordable plot,
    one plot per move. No E/A tradeoff.
    """
    if unclaimed.size == 0 or B_G_remain <= 1e-12:
        return None
    afford = unclaimed[A[unclaimed] <= B_G_remain + 1e-12]
    if afford.size == 0:
        return None
    i = afford[np.argmax(E[afford])]
    return int(i)

def green_pick_max_efficiency_budget(E: np.ndarray,
                                     A: np.ndarray,
                                     unclaimed: np.ndarray,
                                     B_G_remain: float,
                                     theta: float,
                                     lam_t: Optional[float]) -> Optional[int]:
    """
    'Max Efficiency' under a budget: pick by E - (lambda_t + theta) * A among affordable plots.
    This is the cost-effectiveness (E/A) flavored rule.
    """
    if unclaimed.size == 0 or B_G_remain <= 1e-12:
        return None

    # Compute lambda_t if not provided
    if lam_t is None:
        lam_t = compute_lambda_for_budget(E, A, unclaimed, B_G_remain, theta=theta)

    # Score among affordable items
    afford_mask = A[unclaimed] <= B_G_remain + 1e-12
    if not np.any(afford_mask):
        return None
    cand = unclaimed[afford_mask]
    score = E[cand] - (lam_t + theta) * A[cand]

    # tolerant threshold near the frontier
    _eps = 1e-9
    pos = score >= -_eps
    if not np.any(pos):
        return None

    i = cand[np.argmax(score)]
    return int(i)



def green_pick_block_budget(E: np.ndarray,
                            A: np.ndarray,
                            unclaimed: np.ndarray,
                            B_G_remain: float) -> Optional[int]:
    """
    Pure threat-based budgeted picker: buy the highest-A affordable item.
    """
    if unclaimed.size == 0 or B_G_remain <= 1e-12:
        return None
    afford = unclaimed[A[unclaimed] <= B_G_remain + 1e-12]
    if afford.size == 0:
        return None
    i = afford[np.argmax(A[afford])]
    return int(i)


def green_pick_hotspot_budget(E: np.ndarray,
                              A: np.ndarray,
                              unclaimed: np.ndarray,
                              B_G_remain: float,
                              use_product: bool = True,
                              theta_add: float = 0.0) -> Optional[int]:
    """
    Hot-spot under budget: either product E*A or additive E + theta_add*A (both common).
    """
    if unclaimed.size == 0 or B_G_remain <= 1e-12:
        return None
    afford = unclaimed[A[unclaimed] <= B_G_remain + 1e-12]
    if afford.size == 0:
        return None
    if use_product:
        score = E[afford] * A[afford]
    else:
        score = E[afford] + theta_add * A[afford]
    i = afford[np.argmax(score)]
    return int(i)

def simulate_game_budget(E: np.ndarray,
                         A: np.ndarray,
                         green_rule: str,
                         farmer_strategy: str,
                         leakage: float,
                         B_G_init: float | None,
                         rounds: int,  # kept for API compatibility; loop stops on spend/grid-exhaustion
                         rng: Optional[np.random.Generator] = None,
                         theta: float = 0.0,
                         hotspot_additive: bool = False,
                         multi_per_round: bool = False,
                         farmer_share: float | None = 0.5,           # coverage share (default 50/50)
                         use_claims_consistent_farmer: bool = False, # alt coverage via claims count
                         claims_farmer_override: int | None = None   # custom claims count if desired
                         ) -> Dict[str, Any]:
    """
    Budgeted sequential contest (final):
      - Prices = A; budgets calibrated per grid.
      - Farmer 'strategic' pre-empts risky high-E plots (those Greens could afford under BAU) before safe ones.
      - Stop when budgets spent OR grid exhausted (≤ ~50 cycles).
      - Purchased Conservation only (no residual credit) unless (coverage & leakage<1 & true burn>0) ⇒ DLE credit.
      - Event-based additionality (EA) maintained: +E when Greens take Farmer-BAU; −E when Farmer takes Green-BAU.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = E.size
    all_idx = np.arange(n)

    # === 0) tiny local helpers (budget mode) =================================
    def _green_affordable_by_E_indices(E_: np.ndarray, A_: np.ndarray, B_G_: float) -> List[int]:
        """Greedy Max-E purchase list under budget B_G_ (no opposition)."""
        order = np.argsort(-E_)
        picked, spend = [], 0.0
        for i in order:
            ai = float(A_[i])
            if spend + ai <= B_G_:
                picked.append(int(i))
                spend += ai
        return picked

    def _farmer_bau_budget_indices_strategic(E_: np.ndarray, A_: np.ndarray,
                                             B_F_: float, B_G_: float) -> List[int]:
        """
        Strategic Farmer BAU (budget): pre-empt 'risky' ≈ plots Greens could afford by Max-E under B_G_,
        picking within risky by descending A until B_F_ is binding; then fill safe by descending A.
        """
        risky_set = set(_green_affordable_by_E_indices(E_, A_, B_G_))
        risky = np.array(sorted(list(risky_set), key=lambda i: -A_[i]), dtype=int)
        picked: List[int] = []
        spend = 0.0
        # take risky first (highest A among risky)
        for i in risky:
            ai = float(A_[i])
            if spend + ai <= B_F_:
                picked.append(int(i))
                spend += ai
        # then safe by A
        not_picked_mask = np.ones(n, dtype=bool)
        if picked:
            not_picked_mask[np.array(picked, dtype=int)] = False
        safe = np.where(not_picked_mask)[0]
        # remove any risky already considered (keep remaining unpicked)
        if risky.size > 0:
            safe = np.setdiff1d(safe, risky, assume_unique=False)
        safe_order = safe[np.argsort(-A_[safe])]
        for i in safe_order:
            ai = float(A_[i])
            if spend + ai <= B_F_:
                picked.append(int(i))
                spend += ai
        return picked

    # === 1) Calibrate budgets per grid =======================================
    if use_claims_consistent_farmer:
        cF = int(claims_farmer_override) if claims_farmer_override is not None else (n // 2)
        B_F_init, B_G_cov, A_tot, is_coverage = calibrate_budgets(
            A, claims_farmer=cF, farmer_share=None, B_G_override=None
        )
    else:
        B_F_init, B_G_cov, A_tot, is_coverage = calibrate_budgets(
            A, claims_farmer=None, farmer_share=farmer_share, B_G_override=None
        )
    if B_G_init is not None:
        B_F_init, B_G_cov, A_tot, is_coverage = calibrate_budgets(
            A, claims_farmer=None, farmer_share=None, B_G_override=B_G_init
        )

    # === 2) BAU sets for bookkeeping (budget analogues) ======================
    # Farmer BAU under budgets (depends on farmer_strategy)
    if farmer_strategy.lower() == 'strategic':
        farmer_bau_budget = _farmer_bau_budget_indices_strategic(E, A, B_F_init, B_G_cov)
    else:  # 'naive'
        farmer_bau_budget = farmer_bau_budget_indices(A, B_F_init)

    farmer_bau_mask = np.zeros(n, dtype=bool); farmer_bau_mask[farmer_bau_budget] = True

    # Strategic Farmer "risky" mask (for dynamic play): same risky definition as above
    risky_mask = np.zeros(n, dtype=bool)
    if farmer_strategy.lower() == 'strategic':
        risky_indices = _green_affordable_by_E_indices(E, A, B_G_cov)
        risky_mask[np.array(risky_indices, dtype=int)] = True

    # Green BAU benchmark under budgets (strategy-invariant Max-E on complement within B_G_cov)
    rem_for_green_bau = np.setdiff1d(all_idx, np.array(farmer_bau_budget, dtype=int), assume_unique=False)
    green_bau_budget = green_bau_budget_indices(E, A, rem_for_green_bau, B_G_cov)
    green_bau_mask = np.zeros(n, dtype=bool); green_bau_mask[green_bau_budget] = True
    greens_bau_score = float(E[np.array(green_bau_budget, dtype=int)].sum())

    # === 3) State =============================================================
    unclaimed = all_idx.copy()
    Sg: List[int] = []
    Sf: List[int] = []

    B_F = float(B_F_init)
    B_G = float(B_G_cov)

    addl_event = 0.0                   # Event Additionality
    farmer_burn_accum = 0.0            # true burn only
    opt_total = optimal_total(E, A)

    rounds_list: List[int] = []
    pc_series:   List[float] = []      # Purchased Conservation time path
    addl_series: List[float] = []
    wgap_series: List[float] = []

    def record_round(rnum: int):
        pc_now = float(E[Sg].sum()) if Sg else 0.0
        welfare_actual = float(A[Sf].sum() + E[Sg].sum())
        welfare_gap = 100.0 * (opt_total - welfare_actual) / opt_total
        rounds_list.append(rnum)
        pc_series.append(pc_now)
        addl_series.append(addl_event)
        wgap_series.append(welfare_gap)

    # === 4) Green rule switch ================================================
    def green_pick_one(uncl: np.ndarray, B_G_remain: float) -> Optional[int]:
        rule = green_rule.lower()
        if rule == 'maxe_budget':  # PURE Max-E (claims analogue)
            return green_pick_maxE_pure_budget(E, A, uncl, B_G_remain)
        elif rule == 'max_efficiency_budget':  # cost-effectiveness (E - lambda*A)
            lam_t = compute_lambda_for_budget(E, A, uncl, B_G_remain, theta=theta)
            return green_pick_max_efficiency_budget(E, A, uncl, B_G_remain, theta=theta, lam_t=lam_t)
        elif rule == 'block_budget':
            return green_pick_block_budget(E, A, uncl, B_G_remain)
        elif rule == 'hotspot_budget':
            return green_pick_hotspot_budget(E, A, uncl, B_G_remain,
                                             use_product=not hotspot_additive,
                                             theta_add=theta)
        else:
            raise ValueError("Unknown green_rule (use maxE_budget|max_efficiency_budget|block_budget|hotspot_budget)")

    # === 5) Main loop: stop when budgets spent OR grid exhausted ============
    r = 0
    max_cycles = 2 * n + 10  # hard safety cap
    while True:
        if unclaimed.size == 0:
            break
        if (B_G <= 1e-12 and B_F <= 1e-12):
            break
        if r >= max_cycles:
            print("[warn] budget loop hit safety cap; breaking")
            break

        r += 1
        farmer_moves = 1
        green_moves  = 1
        purchased_this_round = False

        # FARMER move
        for _ in range(farmer_moves):
            if unclaimed.size == 0 or B_F <= 1e-12:
                break
            afford = unclaimed[A[unclaimed] <= B_F + 1e-12]
            if afford.size == 0:
                break

            if farmer_strategy.lower() == 'strategic':
                # try risky affordable first (argmax A), else fallback to overall argmax A
                risky_afford = afford[risky_mask[afford]]
                if risky_afford.size > 0:
                    f_pick = risky_afford[np.argmax(A[risky_afford])]
                else:
                    f_pick = afford[np.argmax(A[afford])]
            else:
                f_pick = afford[np.argmax(A[afford])]

            Sf.append(int(f_pick))
            if green_bau_mask[int(f_pick)]:  # Farmer takes Green-BAU budget plot → -E
                addl_event -= float(E[int(f_pick)])
            B_F -= float(A[int(f_pick)])
            unclaimed = unclaimed[unclaimed != int(f_pick)]
            purchased_this_round = True

        # GREEN move
        for _ in range(green_moves):
            if unclaimed.size == 0 or B_G <= 1e-12:
                break
            g_pick = green_pick_one(unclaimed, B_G)
            if g_pick is None:
                break
            Sg.append(int(g_pick))
            if farmer_bau_mask[int(g_pick)]:  # Green takes Farmer-BAU → +E and possible budget burn
                addl_event += float(E[int(g_pick)])
                burn = (1.0 - leakage) * float(A[int(g_pick)])
                if burn > 0.0:
                    B_F = max(0.0, B_F - burn)
                    farmer_burn_accum += burn
            B_G -= float(A[int(g_pick)])
            unclaimed = unclaimed[unclaimed != int(g_pick)]
            purchased_this_round = True

        if not purchased_this_round:
            break

        record_round(r)

    # === 6) Endgame: report residuals; credit DLE only under coverage+burn+L<1 ===
    residual_idx = unclaimed.copy()
    residual_E   = float(E[residual_idx].sum()) if residual_idx.size else 0.0
    residual_ct  = int(residual_idx.size)

    # Purchased totals (no residual credit by default)
    cons_purchased_final = float(E[Sg].sum())        # == last pc_series value
    addl_final_event     = float(addl_event)

    # ---- Purchased-only welfare (this is the time-series notion you already plot) ----
    welfare_actual_purch = float(A[Sf].sum() + E[Sg].sum())
    welfare_gap_purch    = 100.0 * (opt_total - welfare_actual_purch) / opt_total
    # (we keep the 'wgap_series' as purchased-only throughout the game)
    # NOTE: we do NOT append an extra final datapoint; last element of wgap_series is purchased-only.

    # ---- Optional DLE credit for "final" (claims-like) outcomes in coverage runs ----
    credit_dle = (bool(is_coverage) and (leakage < 1.0 - 1e-12) and (farmer_burn_accum > 1e-9))

    if credit_dle and residual_ct > 0:
        # Conservation/additionality finals (mirror claims)
        cons_with_dle = cons_purchased_final + residual_E
        res_bau = residual_idx[farmer_bau_mask[residual_idx]] if residual_ct > 0 else np.array([], dtype=int)
        addl_final_event += float(E[res_bau].sum()) if res_bau.size > 0 else 0.0

        # *** Final welfare (incl. DLE): give residual E to Greens in welfare too (like claims) ***
        welfare_actual_final = welfare_actual_purch + residual_E
        cons_final = cons_with_dle
        dle_final  = cons_with_dle - cons_purchased_final
    else:
        # No DLE credit (non-coverage or no burn or L=1): finals == purchased
        welfare_actual_final = welfare_actual_purch
        cons_final = cons_purchased_final
        dle_final  = 0.0

    welfare_gap_final = 100.0 * (opt_total - welfare_actual_final) / opt_total

    # === 7) Return ===
    return dict(
        Sg=Sg, Sf=Sf,
        farmer_bau=farmer_bau_budget, green_bau=green_bau_budget,
        rounds=rounds_list,
        # Purchased-only trajectories (unchanged)
        pse=pc_series,                      # Purchased Conservation (PC) trajectory
        conservation=pc_series,             # alias for plotting
        additionality=addl_series, welfare_gap=wgap_series,
        # Finals:
        pse_final=cons_purchased_final,     # purchased conservation (no residual)
        dle_final=dle_final,                # credited only under guard
        cons_final=cons_final,
        addl_final=addl_final_event,
        welfare_gap_final=welfare_gap_final,    # *** FINAL welfare (incl. DLE when credited) ***
        # Diagnostics:
        farmer_budget_burn=float(farmer_burn_accum),            # true burn only
        farmer_budget_spent=float(B_F_init - max(B_F, 0.0)),    # total spent
        green_budget_spent=float(B_G_cov - max(B_G, 0.0)),
        A_tot=float(A_tot),
        is_coverage=bool(is_coverage),
        residual_count=residual_ct,
        residual_E=residual_E,
        leakage=float(leakage),
    )



def run_replications_budget(green_rule: str,
                            farmer_strategy: str,
                            leakage: float,
                            rho: float,
                            grid: int,
                            rounds: int,
                            B_G: float | None,
                            reps: int,
                            seed: Optional[int] = 42,
                            theta: float = 0.0,
                            hotspot_additive: bool = False,
                            multi_per_round: bool = False,
                            # passthroughs for simulate_game_budget:
                            farmer_share: float | None = 0.5,
                            use_claims_consistent_farmer: bool = False,
                            claims_farmer_override: int | None = None
                            ) -> Dict[str, Any]:
    rng_master = np.random.default_rng(seed)

    # finals
    final_cons, final_pse, final_dle, final_addl, final_wgap = [], [], [], [], []

    # “pure” (pre-residual) finals
    pure_addl_list, pure_wgap_list = [], []

    # diagnostics
    fburn_list, res_ct_list, res_E_list = [], [], []
    g_spent_list, f_spent_list, A_tot_list, coverage_flags = [], [], [], []

    # time series (for dynamic overlays)
    longest = 0
    tse_pse, tse_cons, tse_addl, tse_wgap = [], [], [], []

    for _ in range(reps):
        E, A = initialize_grid(n=grid, rho=rho, rng=rng_master)

        res = simulate_game_budget(
            E, A,
            green_rule=green_rule,
            farmer_strategy=farmer_strategy,
            leakage=leakage,
            B_G_init=B_G,
            rounds=rounds,
            rng=rng_master,
            theta=theta,
            hotspot_additive=hotspot_additive,
            multi_per_round=multi_per_round,
            farmer_share=farmer_share,
            use_claims_consistent_farmer=use_claims_consistent_farmer,
            claims_farmer_override=claims_farmer_override
        )

        # finals
        final_cons.append(res['cons_final'])
        final_pse.append(res['pse_final'])
        final_dle.append(res['dle_final'])
        final_addl.append(res['addl_final'])
        final_wgap.append(res['welfare_gap_final'])

        # pure (pre-residual) = last points of the series
        pure_addl_list.append(res['additionality'][-1] if res['additionality'] else 0.0)
        pure_wgap_list.append(res['welfare_gap'][-1] if res['welfare_gap'] else 0.0)

        # diagnostics
        fburn_list.append(res.get('farmer_budget_burn', 0.0))
        res_ct_list.append(res.get('residual_count', 0))
        res_E_list.append(res.get('residual_E', 0.0))
        g_spent_list.append(res.get('green_budget_spent', 0.0))
        f_spent_list.append(res.get('farmer_budget_spent', 0.0))
        A_tot_list.append(res.get('A_tot', float(np.sum(A))))
        coverage_flags.append(1.0 if res.get('is_coverage', False) else 0.0)

        # trajectories
        longest = max(longest, len(res['rounds']))
        tse_pse.append(np.array(res['pse'], dtype=float))            # PC path
        tse_cons.append(np.array(res['conservation'], dtype=float))  # alias
        tse_addl.append(np.array(res['additionality'], dtype=float))
        tse_wgap.append(np.array(res['welfare_gap'], dtype=float))

    def pad_and_mean(series_list: List[np.ndarray]) -> np.ndarray:
        padded = []
        for arr in series_list:
            if arr.size == 0:
                padded.append(np.zeros(longest))
            elif arr.size < longest:
                last = arr[-1]
                pad = np.full(longest - arr.size, last)
                padded.append(np.concatenate([arr, pad]))
            else:
                padded.append(arr[:longest])
        return np.mean(np.vstack(padded), axis=0)

    return dict(
        # dynamics
        mean_pse_ts=pad_and_mean(tse_pse),
        mean_cons_ts=pad_and_mean(tse_cons),
        mean_addl_ts=pad_and_mean(tse_addl),
        mean_wgap_ts=pad_and_mean(tse_wgap),
        # finals (with DLE if credited by engine guard)
        mean_cons=float(np.mean(final_cons)),
        mean_pse=float(np.mean(final_pse)),
        mean_dle=float(np.mean(final_dle)),
        mean_addl=float(np.mean(final_addl)),
        mean_wgap=float(np.mean(final_wgap)),
        # pure (pre-residual) finals
        mean_addl_pure=float(np.mean(pure_addl_list)),
        mean_wgap_pure=float(np.mean(pure_wgap_list)),
        # diagnostics
        mean_fburn=float(np.mean(fburn_list)),
        mean_res_ct=float(np.mean(res_ct_list)),
        mean_res_E=float(np.mean(res_E_list)),
        mean_g_spent=float(np.mean(g_spent_list)),
        mean_f_spent=float(np.mean(f_spent_list)),
        mean_A_tot=float(np.mean(A_tot_list)),
        coverage_rate=float(np.mean(coverage_flags)),
    )



# --------------------------
# Replications and summaries
# --------------------------

def run_replications(green_strategy: str,
                     farmer_strategy: str,
                     leakage: float,
                     rho: float,
                     grid: int,
                     rounds: int,
                     claims_green: int,
                     claims_farmer: int,
                     reps: int,
                     seed: Optional[int] = 42,
                     multi_per_round: bool = False,
                     risky_rule: str = 'green_claims'
                     ) -> Dict[str, object]:
    """
    Run multiple replications, generating a new grid each time, and average the time series and final outcomes.
    """
    rng_master = np.random.default_rng(seed)
    final_cons, final_pse, final_dle, final_addl, final_wgap = [], [], [], [], []

    longest = 0
    tse_pse, tse_cons, tse_addl, tse_wgap = [], [], [], []

    for _ in range(reps):
        E, A = initialize_grid(n=grid, rho=rho, rng=rng_master)
        res = simulate_game_dynamic_BAU(
            E, A, green_strategy, farmer_strategy, leakage,
            claims_green, claims_farmer, rounds, rng_master,
            multi_per_round=multi_per_round, risky_rule=risky_rule
        )
        final_cons.append(res['cons_final'])
        final_pse.append(res['pse_final'])
        final_dle.append(res['dle_final'])
        final_addl.append(res['addl_final'])
        final_wgap.append(res['welfare_gap_final'])

        longest = max(longest, len(res['rounds']))
        tse_pse.append(np.array(res['pse'], dtype=float))
        tse_cons.append(np.array(res['conservation'], dtype=float))
        tse_addl.append(np.array(res['additionality'], dtype=float))
        tse_wgap.append(np.array(res['welfare_gap'], dtype=float))

    def pad_and_mean(series_list: List[np.ndarray]) -> np.ndarray:
        padded = []
        for arr in series_list:
            if arr.size == 0:
                padded.append(np.zeros(longest))
            elif arr.size < longest:
                last = arr[-1]
                pad = np.full(longest - arr.size, last)
                padded.append(np.concatenate([arr, pad]))
            else:
                padded.append(arr[:longest])
        return np.mean(np.vstack(padded), axis=0)

    return dict(
        mean_pse_ts=pad_and_mean(tse_pse),
        mean_cons_ts=pad_and_mean(tse_cons),
        mean_addl_ts=pad_and_mean(tse_addl),
        mean_wgap_ts=pad_and_mean(tse_wgap),
        mean_cons=np.mean(final_cons),
        mean_pse=np.mean(final_pse),
        mean_dle=np.mean(final_dle),
        mean_addl=np.mean(final_addl),
        mean_wgap=np.mean(final_wgap),
    )


# --------------------------
# Effects table (PSE + DLE)
# --------------------------

def compute_effects_table(greens: List[str],
                          farmer_strategy: str,
                          leakages: List[float],
                          rho: float,
                          grid: int,
                          rounds: int,
                          claims_green: int,
                          claims_farmer: int,
                          reps: int,
                          seed: Optional[int] = 42,
                          multi_per_round: bool = False,
                          risky_rule: str = 'green_claims'
                          ) -> pd.DataFrame:
    rows = []
    for g in greens:
        for L in leakages:
            res = run_replications(
                green_strategy=g, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed, multi_per_round=multi_per_round, risky_rule=risky_rule
            )
            rows.append(dict(
                farmer_strategy=farmer_strategy,
                green_strategy=g,
                leakage_pct=int(round(L * 100)),
                pure_strategy_effect=round(res['mean_pse'], 3),
                displacement_leakage_effect=round(res['mean_dle'], 3),
                final_conservation_score=round(res['mean_cons'], 3)
            ))
    df = pd.DataFrame(rows)
    return df[['farmer_strategy','green_strategy','leakage_pct',
               'pure_strategy_effect','displacement_leakage_effect','final_conservation_score']]

# --------------------------
# Plotting 2.0 (CIs + Dynamics, Claims & Budget)
# --------------------------

import colorsys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# ---------- Utilities ----------

def _adjust_lightness(color, factor):
    """
    Return a lighter/darker shade of a matplotlib color.
    factor > 1  => lighter ; factor < 1 => darker
    """
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, 1 - (1 - l) / factor)) if factor >= 1 else max(0, min(1, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)

def _bootstrap_ci_for_mean(values, confidence_level=0.95, boot_iters=1000, rng=None):
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n <= 1:
        m = float(arr.mean() if n == 1 else 0.0)
        return (m, m)
    if rng is None:
        rng = np.random.default_rng()
    means = []
    for _ in range(boot_iters):
        samp = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(samp)))
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(means, (alpha/2.0)*100.0))
    hi = float(np.percentile(means, (1.0 - alpha/2.0)*100.0))
    return lo, hi

def _safe_norm_name(name: str) -> str:
    """Normalize strategy key if helper exists in your script."""
    if 'normalize_green_strategy_name' in globals():
        return normalize_green_strategy_name(name)
    return name

# Base colors per strategy (expandable)
STRAT_COLORS = {
    'max_env':       '#2ca02c',  # green
    'hot_spot':      '#ff7f0e',  # orange
    'block_farmers': '#1f77b4',  # blue
    'max_diff':      '#9467bd',  # purple
    'random':        '#8c564b',  # brown
}

BUDGET_RULE_COLORS = {
    'maxE_budget':           '#2ca02c',  # green
    'hotspot_budget':        '#ff7f0e',  # orange
    'block_budget':          '#1f77b4',  # blue
    'max_efficiency_budget': '#17becf',  # teal
}


# Put this just below BUDGET_RULE_COLORS
BUDGET_RULES_ALL = ['maxE_budget', 'hotspot_budget', 'block_budget', 'max_efficiency_budget']


# Metric maps (engine keys)
_CLAIMS_FINAL_KEYS = {
    'conservation':  ('cons_final',          'Final Conservation'),
    'additionality': ('addl_final',          'Final Additionality (incl. DLE)'),
    'welfare':       ('welfare_gap_final',   'Final Welfare Loss (%)'),
}
_BUDGET_FINAL_KEYS = {
    'conservation':  ('cons_final',          'Final (incl. DLE) Conservation'),
    'additionality': ('addl_final',          'Final (incl. DLE) Additionality'),
    'welfare':       ('welfare_gap_final',   'Final Welfare Loss (%)'),
}
# Dynamic TS keys returned by run_replications / run_replications_budget
_TS_KEYS = {
    'conservation':  ('mean_cons_ts',  'Conservation',                'mean_cons'),
    'additionality': ('mean_addl_ts',  'Additionality',               'mean_addl'),
    'welfare':       ('mean_wgap_ts',  'Welfare Loss (%)',            'mean_wgap'),
}

# ---------- Static with CI (Claims) ----------

def _compute_claims_metric_meanCI(
    greens: List[str], farmer_strategy: str, leakages: List[float], metric: str,
    *, rho: float, grid: int, rounds: int, claims_green: int, claims_farmer: int,
    reps: int, seed: Optional[int], multi_per_round: bool, risky_rule: str,
    confidence_level: float = 0.95, boot_iters: int = 1000
):
    key, _ = _CLAIMS_FINAL_KEYS[metric]
    rng_master = np.random.default_rng(seed)
    out = {}
    for g in greens:
        for L in leakages:
            vals = []
            for _ in range(reps):
                E, A = initialize_grid(n=grid, rho=rho, rng=rng_master)
                res = simulate_game_dynamic_BAU(
                    E, A,
                    green_strategy=g,
                    farmer_strategy=farmer_strategy,
                    leakage=L,
                    claims_green=claims_green,
                    claims_farmer=claims_farmer,
                    rounds=rounds,
                    rng=rng_master,
                    multi_per_round=multi_per_round,
                    risky_rule=risky_rule
                )
                vals.append(res[key])
            vals = np.asarray(vals, dtype=float)
            m  = float(np.mean(vals))
            lo, hi = _bootstrap_ci_for_mean(vals, confidence_level, boot_iters, rng_master)
            out[(g, L)] = dict(mean=m, low=lo, high=hi)
    return out

def _plot_metric_vs_leakage_with_ci(
    out_path: Path,
    title: str,
    series_map: dict,        # {(label, leakage_float) -> {'mean','low','high'}}
    labels_in_order: List[str],
    color_map: dict,
    ylabel: str,
    normalize_labels: bool = False
):
    rows = []
    Lvals = sorted({L for (_, L) in series_map.keys()})
    for lab in labels_in_order:
        for L in Lvals:
            s = series_map[(lab, L)]
            rows.append(dict(
                label=lab,
                leakage=L,
                leakage_pct=int(round(100 * L)),
                mean=s['mean'], low=s['low'], high=s['high']
            ))
    df = pd.DataFrame(rows).sort_values(['label', 'leakage'])

    plt.figure(figsize=(7.2, 5.2))
    for lab in labels_in_order:
        sub = df[df['label'] == lab]
        key = _safe_norm_name(lab) if normalize_labels else lab
        base = color_map.get(key, color_map.get(lab,'C0'))
        plt.plot(sub['leakage_pct'], sub['mean'], marker='o', linewidth=2,
                 color=base, label=lab, zorder=3)
        plt.fill_between(sub['leakage_pct'], sub['low'], sub['high'],
                         color=base, alpha=0.22, zorder=1)

    plt.xlabel('Leakage (%)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, zorder=0)
    plt.xticks([100, 75, 50, 25, 0])
    plt.gca().invert_xaxis()
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_claims_static_all_with_CI(
    outdir: Path,
    greens_selected: List[str],                 # include extras (e.g., max_diff, random) if chosen
    farmer_strategy: str,
    leakages: List[float],
    *,
    rho: float, grid: int, rounds: int,
    claims_green: int, claims_farmer: int,
    reps: int, seed: Optional[int],
    multi_per_round: bool, risky_rule: str,
    confidence_level: float = 0.95, boot_iters: int = 1000,
    normalize_labels: bool = True
):
    outdir = Path(outdir)
    # three metrics, one file each; include every selected green strategy
    metrics = ['conservation', 'additionality', 'welfare']
    file_map = {
        'conservation':  f'claims_static_cons_CI_{farmer_strategy}.png',
        'additionality': f'claims_static_addl_CI_{farmer_strategy}.png',
        'welfare':       f'claims_static_welfare_CI_{farmer_strategy}.png',
    }
    ylabels = {m: _CLAIMS_FINAL_KEYS[m][1] for m in metrics}
    for metric in metrics:
        stats = _compute_claims_metric_meanCI(
            greens_selected, farmer_strategy, leakages, metric,
            rho=rho, grid=grid, rounds=rounds,
            claims_green=claims_green, claims_farmer=claims_farmer,
            reps=reps, seed=seed, multi_per_round=multi_per_round,
            risky_rule=risky_rule, confidence_level=confidence_level, boot_iters=boot_iters
        )
        series_map = {(g, L): stats[(g, L)] for g in greens_selected for L in leakages}
        title = f"{ylabels[metric]} vs Leakage — Claims — Farmer={farmer_strategy}"
        out_path = outdir / file_map[metric]
        _plot_metric_vs_leakage_with_ci(out_path, title, series_map, greens_selected, STRAT_COLORS, ylabel=ylabels[metric], normalize_labels=True)
        print(f"[saved] {out_path}")

# ---------- Static with CI (Budget) ----------

def _compute_budget_metric_meanCI(
    rules: List[str], farmer_strategy: str, leakages: List[float], metric: str,
    *, rho: float, grid: int, rounds: int, B_G: float | None,
    reps: int, seed: Optional[int], theta: float, hotspot_additive: bool,
    multi_per_round: bool, farmer_share: float | None = 0.5,
    confidence_level: float = 0.95, boot_iters: int = 1000
):
    key, _ = _BUDGET_FINAL_KEYS[metric]
    rng_master = np.random.default_rng(seed)
    out = {}
    for rule in rules:
        for L in leakages:
            vals = []
            for _ in range(reps):
                E, A = initialize_grid(n=grid, rho=rho, rng=rng_master)
                res = simulate_game_budget(
                    E, A,
                    green_rule=rule,
                    farmer_strategy=farmer_strategy,
                    leakage=L,
                    B_G_init=B_G,
                    rounds=rounds,
                    rng=rng_master,
                    theta=theta,
                    hotspot_additive=hotspot_additive,
                    multi_per_round=multi_per_round,
                    farmer_share=farmer_share,
                )
                vals.append(res[key])
            vals = np.asarray(vals, dtype=float)
            m = float(np.mean(vals))
            lo, hi = _bootstrap_ci_for_mean(vals, confidence_level, boot_iters, rng_master)
            out[(rule, L)] = dict(mean=m, low=lo, high=hi)
    return out

def plot_budget_static_all_with_CI(
    outdir: Path,
    rules: List[str],
    farmer_strategy: str,
    leakages: List[float],
    *,
    rho: float, grid: int, rounds: int, B_G: float | None,
    reps: int, seed: Optional[int], theta: float,
    hotspot_additive: bool, multi_per_round: bool,
    farmer_share: float | None = 0.5,
    confidence_level: float = 0.95, boot_iters: int = 1000,
    write_compat_copies: bool = True,
    normalize_labels: bool = False,
):
    outdir = Path(outdir)
    metrics = ['conservation', 'additionality', 'welfare']
    file_map_consistent = {
        'conservation': f'budget_static_cons_CI_{farmer_strategy}.png',
        'additionality': f'budget_static_addl_CI_{farmer_strategy}.png',
        'welfare': f'budget_static_welfare_CI_{farmer_strategy}.png',
    }
    legacy_map = {
        'conservation': f'budget_static_overlay_cons_final_{farmer_strategy}.png',
        'additionality': f'budget_static_overlay_addl_final_{farmer_strategy}.png',
        'welfare': f'budget_static_overlay_welfare_final_{farmer_strategy}.png',
    }
    ylabels = {m: _BUDGET_FINAL_KEYS[m][1] for m in metrics}

    for metric in metrics:
        stats = _compute_budget_metric_meanCI(
            rules, farmer_strategy, leakages, metric,
            rho=rho, grid=grid, rounds=rounds, B_G=B_G,
            reps=reps, seed=seed, theta=theta,
            hotspot_additive=hotspot_additive, multi_per_round=multi_per_round,
            farmer_share=farmer_share,
            confidence_level=confidence_level, boot_iters=boot_iters,
        )
        series_map = {(r, L): stats[(r, L)] for r in rules for L in leakages}
        title = f"{ylabels[metric]} vs Leakage — Budget — Farmer={farmer_strategy}"
        out_consistent = outdir / file_map_consistent[metric]
        _plot_metric_vs_leakage_with_ci(
            out_consistent, title, series_map, rules, BUDGET_RULE_COLORS,
            ylabel=ylabels[metric], normalize_labels=normalize_labels
        )
        print(f'[saved] {out_consistent}')
        if write_compat_copies:
            from shutil import copyfile
            out_legacy = outdir / legacy_map[metric]
            copyfile(out_consistent, out_legacy)
            print(f'[saved] {out_legacy} (compatibility copy)')

# ---------- Dynamic Overlays (Claims & Budget) ----------
# Limit dynamic overlays to the core 3 strategies/rules to avoid clutter.

_CORE_GREENS_FOR_DYNAMICS = ['max_env', 'hot_spot', 'block_farmers']

def plot_claims_dynamic_all(outdir: Path,
                            greens_for_dynamic: Optional[List[str]],
                            farmer_strategy: str,
                            leakages: List[float],
                            *,
                            rho: float, grid: int, rounds: int,
                            claims_green: int, claims_farmer: int,
                            reps: int, seed: Optional[int],
                            multi_per_round: bool, risky_rule: str):
    """
    One dynamic figure per outcome (Conservation, Additionality, Welfare) for Claims.
    Curves = strategies (core trio), leakage shown light->dark per strategy.
    """
    outdir = Path(outdir)
    greens = greens_for_dynamic or _CORE_GREENS_FOR_DYNAMICS
    if not leakages:
        return
    results = {}
    ranges = {'conservation':[None,None], 'additionality':[None,None], 'welfare':[None,None]}

    for g in greens:
        for L in leakages:
            res = run_replications(
                green_strategy=g, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed, multi_per_round=multi_per_round, risky_rule=risky_rule
            )
            results[(g,L)] = res
            for metric in ['conservation','additionality','welfare']:
                ts_key, _, mean_key = _TS_KEYS[metric]
                ts = res[ts_key]
                lo, hi = ranges[metric]
                lo = np.min(ts) if lo is None else min(lo, np.min(ts))
                hi = np.max(ts) if hi is None else max(hi, np.max(ts))
                # include final mean in y-range
                if mean_key in res:
                    lo = min(lo, res[mean_key]); hi = max(hi, res[mean_key])
                ranges[metric] = [lo, hi]

    # leakage → lightness
    Ls = sorted(leakages)
    light, dark = 1.8, 0.6
    leak_to_factor = {L: (1.0 if len(Ls)==1 else (light + (dark - light)*(1.0 - L))) for L in Ls}
    middle_L = Ls[len(Ls)//2] if len(Ls) > 1 else Ls[0]

    def _draw_dynamic(metric: str, fname: str, title: str):
        ts_key, ylabel, mean_key = _TS_KEYS[metric]
        y0, y1 = ranges[metric]
        pad = 0.02 * (y1 - y0 + 1e-9)
        fig, ax = plt.subplots(figsize=(7, 5))

        for g in greens:
            base = STRAT_COLORS.get(_safe_norm_name(g), 'C0')
            for L in Ls:
                res = results[(g, L)]
                ts_raw = np.asarray(res[ts_key], dtype=float)

                # -------- Harmonize solid path to exactly `rounds` x-points (PSE path) --------
                # 1) flatten any last-step residual bump so the solid line ends before DLE crediting
                ts_plot = ts_raw.copy()
                if len(ts_plot) >= 2 and (ts_plot[-1] - ts_plot[-2]) > 1e-9:
                    ts_plot[-1] = ts_plot[-2]

                # 2) truncate/pad so solid path ends at x=rounds exactly
                if len(ts_plot) > rounds:
                    ts_plot = ts_plot[:rounds]
                elif len(ts_plot) < rounds:
                    ts_plot = np.pad(ts_plot, (0, rounds - len(ts_plot)), mode='edge')

                x = np.arange(1, rounds + 1)

                # color shade and label (label only at the middle leakage)
                color = _adjust_lightness(base, leak_to_factor[L])
                label = g if L == middle_L else None

                # main trajectory (solid)
                ax.plot(x, ts_plot, color=color, linewidth=2, label=label)

                # -------- Dashed DLE jump from x=rounds to x=rounds+1 (only if a real jump) --------
                if mean_key in res and len(ts_plot) > 0:
                    final_y = float(res[mean_key])
                    if abs(final_y - ts_plot[-1]) > 1e-9:  # draw only when there is an actual jump
                        ax.plot([rounds, rounds + 1],
                                [ts_plot[-1], final_y],
                                color=color, linewidth=2, linestyle='--')
                        ax.scatter([rounds + 1], [final_y],
                                   marker='o', s=28, color=color, edgecolor='none')

        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_xlabel('Round (curves = purchased path; final dot shows incl. DLE)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Strategy legend
        strat_legend = ax.legend(title='Strategy', loc='upper left')
        ax.add_artist(strat_legend)

        # Leakage legend (top-right for welfare to avoid overlap)
        from matplotlib.lines import Line2D
        base_rgb = to_rgb('#444444')
        leakage_handles, leakage_labels = [], []
        for L in sorted(Ls, reverse=True):
            c = _adjust_lightness(base_rgb, leak_to_factor[L])
            leakage_handles.append(Line2D([0], [0], color=c, lw=3))
            leakage_labels.append(f'{int(round(100 * L))}%')
        legend_loc = 'upper right' if metric == 'welfare' else 'lower right'
        ax.legend(leakage_handles, leakage_labels, title='Leakage', loc=legend_loc)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=220)
        plt.close(fig)

    _draw_dynamic('conservation',  f'claims_dyn_overlay_conservation_{farmer_strategy}.png',
                  f'Dynamic Conservation — Claims — Farmer={farmer_strategy}')
    _draw_dynamic('additionality', f'claims_dyn_overlay_additionality_{farmer_strategy}.png',
                  f'Dynamic Additionality — Claims — Farmer={farmer_strategy}')
    _draw_dynamic('welfare',       f'claims_dyn_overlay_welfare_{farmer_strategy}.png',
                  f'Dynamic Welfare Loss — Claims — Farmer={farmer_strategy}')

def plot_budget_dynamic_all(outdir: Path,
                            rules: List[str],
                            farmer_strategy: str,
                            leakages: List[float],
                            *,
                            rho: float, grid: int, rounds: int,
                            B_G: float | None, reps: int, seed: Optional[int],
                            theta: float, hotspot_additive: bool, multi_per_round: bool,
                            farmer_share: float | None = 0.5):
    outdir = Path(outdir)
    if not leakages:
        return
    results = {}
    ranges = {'conservation': [None, None], 'additionality': [None, None], 'welfare': [None, None]}
    cov_flags = []

    for rule in rules:
        for L in leakages:
            res = run_replications_budget(
                green_rule=rule, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds, B_G=B_G,
                reps=reps, seed=seed, theta=theta,
                hotspot_additive=hotspot_additive, multi_per_round=multi_per_round,
                farmer_share=farmer_share,
            )
            results[(rule, L)] = res
            cov_flags.append(res.get('coverage_rate', 1.0))
            for metric in ['conservation', 'additionality', 'welfare']:
                ts_key, _, mean_key = _TS_KEYS[metric]
                ts = res[ts_key]
                lo, hi = ranges[metric]
                lo = np.min(ts) if lo is None else min(lo, np.min(ts))
                hi = np.max(ts) if hi is None else max(hi, np.max(ts))
                if mean_key in res:
                    lo = min(lo, res[mean_key])
                    hi = max(hi, res[mean_key])
                ranges[metric] = [lo, hi]

    is_coverage = (np.mean(cov_flags) >= 0.999)
    Ls = sorted(leakages)
    light, dark = 1.8, 0.6
    leak_to_factor = {L: (1.0 if len(Ls) == 1 else (light + (dark - light) * (1.0 - L))) for L in Ls}
    middle_L = Ls[len(Ls) // 2] if len(Ls) > 1 else Ls[0]

    def _draw_dynamic(metric: str, fname: str, title: str, show_jump: bool):
        ts_key, ylabel, mean_key = _TS_KEYS[metric]
        y0, y1 = ranges[metric]
        pad = 0.02 * (y1 - y0 + 1e-9)
        fig, ax = plt.subplots(figsize=(7, 5))
        for rule in rules:
            base = BUDGET_RULE_COLORS.get(rule, 'C0')
            for L in Ls:
                res = results[(rule, L)]
                ts = res[ts_key]
                x = np.arange(1, ts.size + 1)
                color = _adjust_lightness(base, leak_to_factor[L])
                label = rule if L == middle_L else None
                ax.plot(x, ts, color=color, linewidth=2, label=label)
                if show_jump and is_coverage and (mean_key in res):
                    final_y = res[mean_key]
                    ax.plot([x[-1], x[-1] + 1], [ts[-1], final_y], color=color, linewidth=2, linestyle='--')
                    ax.scatter([x[-1] + 1], [final_y], marker='o', s=28, color=color, edgecolor='none')
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_xlabel('Round (final reflects DLE accounting)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        strat_legend = ax.legend(title='Green Rule', loc='upper left')
        ax.add_artist(strat_legend)
        from matplotlib.lines import Line2D
        base_rgb = to_rgb('#444444')
        leakage_handles, leakage_labels = [], []
        for L in sorted(Ls, reverse=True):
            c = _adjust_lightness(base_rgb, leak_to_factor[L])
            leakage_handles.append(Line2D([0], [0], color=c, lw=3))
            leakage_labels.append(f'{int(round(100 * L))}%')
        legend_loc = 'upper right' if metric == 'welfare' else 'lower right'
        ax.legend(leakage_handles, leakage_labels, title='Leakage', loc=legend_loc)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=220)
        plt.close(fig)

    _draw_dynamic('conservation', f'budget_dyn_overlay_conservation_{farmer_strategy}.png',
                  f'Budgeted Dynamic Conservation — Farmer={farmer_strategy}', show_jump=True)
    _draw_dynamic('additionality', f'budget_dyn_overlay_additionality_{farmer_strategy}.png',
                  f'Budgeted Dynamic Additionality — Farmer={farmer_strategy}', show_jump=True)
    _draw_dynamic('welfare', f'budget_dyn_overlay_welfare_{farmer_strategy}.png',
                  f'Budgeted Dynamic Welfare Loss — Farmer={farmer_strategy}', show_jump=True)


# ---- Budget World: PURCHASED-ONLY static with CI ----

_BUDGET_PURCHASED_KEYS = {
    'conservation':  ('pse',        'Purchased Conservation'),
    'additionality': ('addl_pure',  'Purchased Additionality'),
    'welfare':       ('wgap_pure',  'Purchased Welfare Loss (%)'),
}

def _compute_budget_purchased_meanCI(
    rules: List[str], farmer_strategy: str, leakages: List[float], metric: str,
    *, rho: float, grid: int, rounds: int, B_G: float | None,
    reps: int, seed: Optional[int], theta: float, hotspot_additive: bool,
    multi_per_round: bool, farmer_share: float | None = 0.5,
    confidence_level: float = 0.95, boot_iters: int = 1000
):
    rng_master = np.random.default_rng(seed)
    out = {}
    for rule in rules:
        for L in leakages:
            vals = []
            for _ in range(reps):
                E, A = initialize_grid(n=grid, rho=rho, rng=rng_master)
                res = simulate_game_budget(
                    E, A,
                    green_rule=rule,
                    farmer_strategy=farmer_strategy,
                    leakage=L,
                    B_G_init=B_G,
                    rounds=rounds,
                    rng=rng_master,
                    theta=theta,
                    hotspot_additive=hotspot_additive,
                    multi_per_round=multi_per_round,
                    farmer_share=farmer_share,
                )
                if metric == 'conservation':
                    v = float(res.get('pse_final', 0.0))
                elif metric == 'additionality':
                    series = res.get('additionality', [])
                    v = float(series[-1]) if series else 0.0
                elif metric == 'welfare':
                    series = res.get('welfare_gap', [])
                    v = float(series[-1]) if series else 0.0
                else:
                    raise ValueError(f"unknown metric '{metric}'")
                vals.append(v)
            vals = np.asarray(vals, dtype=float)
            m = float(np.mean(vals))
            lo, hi = _bootstrap_ci_for_mean(vals, confidence_level, boot_iters, rng_master)
            out[(rule, L)] = dict(mean=m, low=lo, high=hi)
    return out

def plot_budget_purchased_all_with_CI(
    outdir: Path,
    rules: List[str],
    farmer_strategy: str,
    leakages: List[float],
    *,
    rho: float, grid: int, rounds: int, B_G: float | None,
    reps: int, seed: Optional[int], theta: float,
    hotspot_additive: bool, multi_per_round: bool,
    farmer_share: float | None = 0.5,
    confidence_level: float = 0.95, boot_iters: int = 1000,
    write_compat_copies: bool = True,
    normalize_labels: bool = False,
):
    outdir = Path(outdir)
    metrics = ['conservation', 'additionality', 'welfare']
    file_map_consistent = {
        'conservation': f'budget_static_cons_purchased_CI_{farmer_strategy}.png',
        'additionality': f'budget_static_addl_purchased_CI_{farmer_strategy}.png',
        'welfare': f'budget_static_welfare_purchased_CI_{farmer_strategy}.png',
    }
    legacy_map = {
        'conservation': f'budget_static_overlay_cons_purchased_{farmer_strategy}.png',
        'additionality': f'budget_static_overlay_addl_purchased_{farmer_strategy}.png',
        'welfare': f'budget_static_overlay_welfare_purchased_{farmer_strategy}.png',
    }
    ylabels = {m: _BUDGET_PURCHASED_KEYS[m][1] for m in metrics}

    for metric in metrics:
        stats = _compute_budget_purchased_meanCI(
            rules, farmer_strategy, leakages, metric,
            rho=rho, grid=grid, rounds=rounds, B_G=B_G,
            reps=reps, seed=seed, theta=theta,
            hotspot_additive=hotspot_additive, multi_per_round=multi_per_round,
            farmer_share=farmer_share,
            confidence_level=confidence_level, boot_iters=boot_iters,
        )
        series_map = {(r, L): stats[(r, L)] for r in rules for L in leakages}
        title = f"{ylabels[metric]} vs Leakage — Budget — Farmer={farmer_strategy}"
        out_consistent = outdir / file_map_consistent[metric]
        _plot_metric_vs_leakage_with_ci(
            out_consistent, title, series_map, rules, BUDGET_RULE_COLORS,
            ylabel=ylabels[metric], normalize_labels=normalize_labels
        )
        print(f'[saved] {out_consistent}')
        if write_compat_copies:
            from shutil import copyfile
            out_legacy = outdir / legacy_map[metric]
            copyfile(out_consistent, out_legacy)
            print(f'[saved] {out_legacy} (compatibility copy)')


# ---------- One-call convenience to build EVERYTHING ----------

def make_all_plots_v2(
    outdir: Path,
    *,
    # Leakage grids (displayed as 100 → 0 via inverted x-axis in the plotters)
    leakages_claims: List[float] = [1.0, 0.75, 0.5, 0.25, 0.0],
    leakages_budget: List[float] = [1.0, 0.75, 0.5, 0.25, 0.0],
    # Simulation controls
    rho: float = 0.0, grid: int = 10, rounds: int = 50,
    reps: int = 500, seed: Optional[int] = 42,
    # Allocation / rules
    alloc: str = 'equal', farmer_pct: float = 0.7,
    risky_rule: str = 'green_claims',
    B_G: float | None = None, theta: float = 0.0,
    hotspot_additive: bool = False,
    # Strategy selections
    greens_selected: Optional[List[str]] = None,    # STATIC (Claims): include extras like max_diff, random if selected
    greens_for_dynamic: Optional[List[str]] = None, # DYNAMICS (Claims): defaults to core trio if None
    # Outputs
    write_effects_csv: bool = True
):
    """
    Generate ALL figures and (optionally) CSV summaries for BOTH farmer types ('naive','strategic'):

      Claims World
        - Static (with bootstrap mean 95% CIs): Conservation, Additionality, Welfare
          * includes ALL strategies in `greens_selected` (e.g., 'max_diff', 'random' if chosen)
        - Dynamic trajectories: Conservation, Additionality, Welfare
          * uses `greens_for_dynamic` or the core trio ['max_env','hot_spot','block_farmers']

      Budget World
        - Static (with bootstrap mean 95% CIs): Conservation, Additionality, Welfare
          * saves to consistent CI names AND legacy overlay names (compatibility copies)
        - Dynamic trajectories: Conservation, Additionality, Welfare

      CSVs (if write_effects_csv=True)
        - claims_effects_table_{farmer}.csv
        - budget_effects_summary_{farmer}.csv
    """
    # Ensure output directory exists
    outdir = ensure_outdir(outdir)

    # Default strategy sets
    if greens_selected is None:
        # Include all known strategies so STATIC plots can show user-selected extras
        greens_selected = ['max_env', 'hot_spot', 'block_farmers', 'max_diff', 'random']

    # Derive claims counts from grid & allocation (shared baseline)
    total_cells = grid * grid
    claims_green, claims_farmer = allocate_points(total_cells, allocation=alloc, farmer_percentage=farmer_pct)

    # Batching logic: Claims uses multi-per-round if user chose fewer rounds than claims;
    # Budget dynamics keep single-move per round unless you later change plotters.
    multi_per_round_claims = (rounds < max(claims_green, claims_farmer))
    multi_per_round_budget = False

    for farmer_strategy in ['naive', 'strategic']:
        # ---- Claims: Static with CI (includes all selected strategies) ----
        plot_claims_static_all_with_CI(
            outdir, greens_selected, farmer_strategy, leakages_claims,
            rho=rho, grid=grid, rounds=rounds,
            claims_green=claims_green, claims_farmer=claims_farmer,
            reps=reps, seed=seed, multi_per_round=multi_per_round_claims,
            risky_rule=risky_rule, confidence_level=0.95, boot_iters=1000
        )

        # ---- Claims: Dynamics (core trio unless overridden) ----
        plot_claims_dynamic_all(
            outdir, greens_for_dynamic, farmer_strategy, leakages_claims,
            rho=rho, grid=grid, rounds=rounds,
            claims_green=claims_green, claims_farmer=claims_farmer,
            reps=reps, seed=seed, multi_per_round=multi_per_round_claims,
            risky_rule=risky_rule
        )

        # ---- Budget: Static with CI (also writes legacy overlay filenames) ----
        plot_budget_static_all_with_CI(
            outdir, BUDGET_RULES_ALL, farmer_strategy, leakages_budget,
            rho=rho, grid=grid, rounds=rounds, B_G=B_G,
            reps=reps, seed=seed, theta=theta,
            hotspot_additive=hotspot_additive, multi_per_round=multi_per_round_budget,
            confidence_level=0.95, boot_iters=1000, write_compat_copies=True
        )
        # ---- Budget: Purchased-only static with CI (also writes legacy overlay filenames) ----
        plot_budget_purchased_all_with_CI(
            outdir, BUDGET_RULES_ALL, farmer_strategy, leakages_budget,
            rho=rho, grid=grid, rounds=rounds, B_G=B_G,
            reps=reps, seed=seed, theta=theta,
            hotspot_additive=hotspot_additive, multi_per_round=multi_per_round_budget,
            confidence_level=0.95, boot_iters=1000, write_compat_copies=True
        )

        # ---- Budget: Dynamics  ----
        plot_budget_dynamic_all(
            outdir, BUDGET_RULES_ALL, farmer_strategy, leakages_budget,
            rho=rho, grid=grid, rounds=rounds, B_G=B_G,
            reps=reps, seed=seed, theta=theta,
            hotspot_additive=hotspot_additive, multi_per_round=multi_per_round_budget
        )

        # ---- CSV exports (optional) ----
        if write_effects_csv:
            export_claims_effects_table(
                outdir, greens_selected, farmer_strategy, leakages_claims,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed, multi_per_round=multi_per_round_claims,
                risky_rule=risky_rule,
                filename=f'claims_effects_table_{farmer_strategy}.csv'
            )
            export_budget_effects_summary(
                outdir, farmer_strategy, BUDGET_RULES_ALL, leakages_budget,
                rho=rho, grid=grid, rounds=rounds, B_G=B_G,
                reps=reps, seed=seed, theta=theta,
                hotspot_additive=hotspot_additive, multi_per_round=multi_per_round_budget,
                filename=f'budget_effects_summary_{farmer_strategy}.csv'
            )



# --------------------------
# CLI
# --------------------------

def _parse_leakages_csv(value: str, fallback: List[float]) -> List[float]:
    try:
        out = [float(x) for x in str(value).split(',') if str(x).strip()]
        out = [x for x in out if 0.0 <= x <= 1.0]
        return out or fallback
    except Exception:
        return fallback


def _selected_dynamic_claims_strategies(greens_selected: List[str]) -> List[str]:
    core = [g for g in _CORE_GREENS_FOR_DYNAMICS if g in greens_selected]
    if core:
        return core
    return greens_selected[:3] if greens_selected else ['max_env', 'hot_spot', 'block_farmers']


def make_claims_world_paper_plots(outdir: Path,
                                  greens_selected: List[str],
                                  farmer_strategy: str,
                                  leakages: List[float],
                                  *,
                                  rho: float, grid: int, rounds: int,
                                  claims_green: int, claims_farmer: int,
                                  reps: int, seed: Optional[int],
                                  risky_rule: str):
    dyn_greens = _selected_dynamic_claims_strategies(greens_selected)

    plot_claims_static_all_with_CI(
        outdir=outdir,
        greens_selected=greens_selected,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds,
        claims_green=claims_green, claims_farmer=claims_farmer,
        reps=reps, seed=seed,
        multi_per_round=False, risky_rule=risky_rule,
        confidence_level=0.95, boot_iters=1000,
    )

    plot_claims_dynamic_all(
        outdir=outdir,
        greens_for_dynamic=dyn_greens,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds,
        claims_green=claims_green, claims_farmer=claims_farmer,
        reps=reps, seed=seed,
        multi_per_round=False, risky_rule=risky_rule,
    )

    export_claims_effects_table(
        outdir=outdir,
        greens_selected=greens_selected,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds,
        claims_green=claims_green, claims_farmer=claims_farmer,
        reps=reps, seed=seed,
        multi_per_round=False, risky_rule=risky_rule,
    )


def make_budget_world_paper_plots(outdir: Path,
                                  rules: List[str],
                                  farmer_strategy: str,
                                  leakages: List[float],
                                  *,
                                  rho: float, grid: int, rounds: int,
                                  B_G: float | None,
                                  reps: int, seed: Optional[int],
                                  theta: float, hotspot_additive: bool,
                                  farmer_share: float | None):
    plot_budget_static_all_with_CI(
        outdir=outdir,
        rules=rules,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds, B_G=B_G,
        reps=reps, seed=seed, theta=theta,
        hotspot_additive=hotspot_additive,
        multi_per_round=False,
        farmer_share=farmer_share,
        confidence_level=0.95, boot_iters=1000,
    )

    plot_budget_purchased_all_with_CI(
        outdir=outdir,
        rules=rules,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds, B_G=B_G,
        reps=reps, seed=seed, theta=theta,
        hotspot_additive=hotspot_additive,
        multi_per_round=False,
        farmer_share=farmer_share,
        confidence_level=0.95, boot_iters=1000,
    )

    plot_budget_dynamic_all(
        outdir=outdir,
        rules=rules,
        farmer_strategy=farmer_strategy,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds, B_G=B_G,
        reps=reps, seed=seed, theta=theta,
        hotspot_additive=hotspot_additive,
        multi_per_round=False,
        farmer_share=farmer_share,
    )

    export_budget_effects_summary(
        outdir=outdir,
        farmer_strategy=farmer_strategy,
        rules=rules,
        leakages=leakages,
        rho=rho, grid=grid, rounds=rounds, B_G=B_G,
        reps=reps, seed=seed, theta=theta,
        hotspot_additive=hotspot_additive,
        multi_per_round=False,
        farmer_share=farmer_share,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Conservation Strategy Game — Replication Package')
    ap.add_argument('--world', choices=['claims', 'budget'], default='claims',
                    help='Choose which world to run.')
    ap.add_argument('--mode', default='all', help=argparse.SUPPRESS)  # legacy; ignored
    ap.add_argument('--farmer', choices=['naive', 'strategic'], nargs='+',
                    default=['naive', 'strategic'],
                    help='One or more farmer strategies to run.')
    ap.add_argument('--greens', default='max_env,hot_spot,block_farmers,max_diff,random',
                    help='Claims World green strategies (comma-separated).')
    ap.add_argument('--budget_rules', default='maxE_budget,hotspot_budget,block_budget,max_efficiency_budget',
                    help='Budget World green rules (comma-separated).')
    ap.add_argument('--leakages', default='1.0,0.75,0.5,0.25,0.0',
                    help='Comma-separated leakage values in [0,1].')
    ap.add_argument('--rho', type=float, default=0.0, help='Correlation between E and A.')
    ap.add_argument('--grid', type=int, default=10, help='Grid size n (n x n cells).')
    ap.add_argument('--rounds', default='auto', help='Use "auto" or provide an integer.')
    ap.add_argument('--reps', type=int, default=500, help='Monte Carlo replications.')
    ap.add_argument('--alloc', choices=['equal', 'political'], default='equal',
                    help='Equal split or political split in favour of Farmers.')
    ap.add_argument('--farmer_pct', type=float, default=0.7,
                    help='Farmer share if allocation is political.')
    ap.add_argument('--risky-rule', choices=['green_claims', 'farmer_claims'], default='green_claims',
                    help='Claims World strategic-farmer risky-set rule.')
    ap.add_argument('--seed', type=int, default=42, help='Random seed.')
    ap.add_argument('--outdir', default='outputs_spatial', help='Directory for figures/tables.')

    # advanced budget options retained for CLI use
    ap.add_argument('--budget_theta', type=float, default=0.0, help='Advanced Budget World option.')
    ap.add_argument('--hotspot_additive', choices=['on', 'off'], default='off',
                    help='Advanced Budget World option.')
    ap.add_argument('--B_G', type=float, default=None,
                    help='Advanced Budget World override for Green budget. Leave blank for coverage shares.')
    ap.add_argument('--use_batching_budget', choices=['on', 'off'], default='off',
                    help=argparse.SUPPRESS)

    return ap.parse_args()


def main():
    args = parse_args()

    if USE_UI_DEFAULT and _gui_available():
        ui = launch_ui_and_get_args()
        if ui:
            args.world = ui['world']
            args.reps = ui['reps']
            args.rho = ui['rho']
            args.alloc = ui['alloc']
            args.farmer_pct = ui['farmer_pct']
            args.greens = ui['claims_greens']
            args.budget_rules = ui['budget_rules']
            args.farmer = ui['farmer']
            args.leakages = ui['leakages']
            args.grid = ui['grid']
            args.rounds = ui['rounds']
            args.outdir = ui['outdir']
            args.seed = ui['seed']
        else:
            print('[info] UI cancelled or returned no values; using CLI arguments.')
    elif USE_UI_DEFAULT:
        print('[info] UI not available in this environment; using CLI arguments.')

    if args.rho < -1.0 or args.rho > 1.0:
        print(f'[warn] rho={args.rho} out of range [-1,1]; clipping.')
        args.rho = max(-1.0, min(1.0, args.rho))

    outdir = ensure_outdir(args.outdir)

    greens_selected = [normalize_green_strategy_name(s) for s in str(args.greens).split(',') if str(s).strip()]
    if not greens_selected:
        greens_selected = ['max_env', 'hot_spot', 'block_farmers', 'max_diff', 'random']

    budget_rules = [s.strip() for s in str(args.budget_rules).split(',') if str(s).strip()]
    if not budget_rules:
        budget_rules = list(BUDGET_RULES_ALL)

    leakages = _parse_leakages_csv(args.leakages, [1.0, 0.75, 0.5, 0.25, 0.0])

    farmer_list = list(args.farmer) if isinstance(args.farmer, (list, tuple)) else [args.farmer]
    if not farmer_list:
        farmer_list = ['naive', 'strategic']

    total_cells = args.grid * args.grid
    claims_green, claims_farmer = allocate_points(
        total_cells,
        allocation=args.alloc,
        farmer_percentage=args.farmer_pct
    )

    if isinstance(args.rounds, str) and args.rounds.strip().lower() == 'auto':
        computed_rounds = max(claims_green, claims_farmer)
    else:
        try:
            computed_rounds = int(args.rounds)
            if computed_rounds <= 0:
                raise ValueError
        except Exception:
            print(f'[warn] rounds={args.rounds} invalid; falling back to auto.')
            computed_rounds = max(claims_green, claims_farmer)

    farmer_share = args.farmer_pct if args.alloc == 'political' else 0.5

    print('[info] '
          f'world={args.world}, '
          f'farmers={farmer_list}, '
          f'leakages={leakages}, '
          f'grid={args.grid}, claims=(G={claims_green}, F={claims_farmer}), '
          f'rounds={computed_rounds}, rho={args.rho}, reps={args.reps}, seed={args.seed}')

    if args.world == 'claims':
        for farmer_strategy in farmer_list:
            make_claims_world_paper_plots(
                outdir=outdir,
                greens_selected=greens_selected,
                farmer_strategy=farmer_strategy,
                leakages=leakages,
                rho=args.rho,
                grid=args.grid,
                rounds=computed_rounds,
                claims_green=claims_green,
                claims_farmer=claims_farmer,
                reps=args.reps,
                seed=args.seed,
                risky_rule=args.risky_rule,
            )
    else:
        if args.B_G is not None:
            print('[info] B_G override supplied; this will override equal/political share calibration in Budget World.')
        for farmer_strategy in farmer_list:
            make_budget_world_paper_plots(
                outdir=outdir,
                rules=budget_rules,
                farmer_strategy=farmer_strategy,
                leakages=leakages,
                rho=args.rho,
                grid=args.grid,
                rounds=computed_rounds,
                B_G=args.B_G,
                reps=args.reps,
                seed=args.seed,
                theta=args.budget_theta,
                hotspot_additive=(args.hotspot_additive == 'on'),
                farmer_share=farmer_share,
            )

    print('[done]')


if __name__ == '__main__':
    main()