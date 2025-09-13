
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conservation Strategy Game — Replication-Ready Simulator
=========================================================

Author: Diana Weinhold
Contact: d.weinhold@lse.ac.uk
All rights reserved.

If you use this simulator in teaching, research, or publications, please cite:

Weinhold, D. & Andersen, L. E. (2025).
"Conservation Strategies in Contested Environments: Dynamic Monte Carlo Simulations and a Bolivian Case Study."
London School of Economics & Universidad Privada Boliviana. Working Paper.

DOI: 10.5281/zenodo.xxxxxxx   # (replace with your actual DOI once minted)
"""


"""
Conservation Strategy Game — Replication-Ready Simulator (v2)
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

USAGE EXAMPLES
--------------
# 1) All outputs for Naive and Strategic farmers (baseline settings, one claim/round)
python conservation_game_sim.py --mode all --farmer naive --farmer strategic

# 2) Static final outcomes (lines vs leakage) for Strategic farmers
python conservation_game_sim.py --mode static --farmer strategic --leakages 1.0,0.5,0.0

# 3) Dynamic trajectories for Naive farmers (multi-claims per round on)
python conservation_game_sim.py --mode dynamic --farmer naive --leakages 1.0,0.5,0.0 \
  --multi-per-round on

# 4) Effects table (PSE, DLE, Final) for both farmer types
python conservation_game_sim.py --mode effects --farmer naive --farmer strategic

# 5) Match a "modern" behavior: multi-claims per round + risky keyed to farmer claims
python conservation_game_sim.py --mode all --farmer strategic \
  --multi-per-round on --risky-rule farmer_claims

# 6) Change correlation (rho), reps, grid size, rounds, and power tilt
python conservation_game_sim.py --mode all --farmer naive \
  --rho 0.0 --reps 500 --grid 10 --rounds 50 \
  --alloc equal --farmer_pct 0.7 --seed 42

REQUIREMENTS
------------
pip install numpy pandas matplotlib pyyaml


OUTPUTS
-------
Figures (*.png) and tables (*.csv) saved into ./outputs/ by default.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# User Interface
# --------------------------

def launch_ui_and_get_args():
    import tkinter as tk
    from tkinter import ttk

    # defaults shown in the UI
    vals = {
        'reps': 500,
        'rho': 0.0,
        'alloc': 'equal',
        'farmer_pct': 0.7,
        'greens': ['max_env','hot_spot','block_farmers'],
        'farmers': ['naive','strategic'],
        'leakages': '1.0,0.5,0.0',
        'grid': 10,
        'rounds': 'auto',
        'outdir': 'outputs',
        'mode': 'all',
        # risky_rule is intentionally not exposed; default = paper rule:
        'risky_rule': 'green_claims',
        'seed': 42,
    }

    root = tk.Tk()
    root.title("Conservation Strategy Game — Settings")

    # consistent padding
    PADX, PADY = 8, 6

    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")
    for c in range(6):  # allow some columns to stretch for spacing rows
        frm.columnconfigure(c, weight=0)

    # ----------------- row 0: Replications -----------------
    ttk.Label(frm, text="Replications").grid(row=0, column=0, sticky="w", padx=PADX, pady=PADY)
    reps_var = tk.IntVar(value=vals['reps'])
    ttk.Spinbox(frm, from_=10, to=100000, increment=10, textvariable=reps_var,
                width=10).grid(row=0, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 1: Correlation ------------------
    ttk.Label(frm, text="Correlation (ρ)").grid(row=1, column=0, sticky="w", padx=PADX, pady=PADY)
    rho_var = tk.DoubleVar(value=vals['rho'])
    tk.Scale(frm, from_=-1.0, to=1.0, resolution=0.1, orient="horizontal",
             variable=rho_var, length=240).grid(row=1, column=1, columnspan=2,
                                                sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 2: Allocation -------------------
    ttk.Label(frm, text="Allocation").grid(row=2, column=0, sticky="w", padx=PADX, pady=PADY)
    alloc_var = tk.StringVar(value=vals['alloc'])
    alloc_cb = ttk.Combobox(frm, values=['equal','political'], textvariable=alloc_var,
                            width=12, state="readonly")
    alloc_cb.grid(row=2, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 3: Farmer % (shown only if political) ------------
    farmer_pct_row = ttk.Frame(frm)
    farmer_pct_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=PADX, pady=PADY)
    ttk.Label(farmer_pct_row, text="Farmer % (if political)").grid(row=0, column=0, sticky="w")
    farmer_pct_var = tk.DoubleVar(value=vals['farmer_pct'])
    tk.Scale(farmer_pct_row, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
             variable=farmer_pct_var, length=240).grid(row=0, column=1, sticky="w", padx=(PADX,0))

    def _toggle_farmer_pct(*_):
        if alloc_var.get() == 'political':
            farmer_pct_row.grid()      # show
        else:
            farmer_pct_row.grid_remove()  # hide

    alloc_cb.bind("<<ComboboxSelected>>", _toggle_farmer_pct)
    _toggle_farmer_pct()  # initial state

    # ----------------- row 4: Green strategies (even spacing) ---------------
    ttk.Label(frm, text="Green strategies").grid(row=4, column=0, sticky="w", padx=PADX, pady=PADY)
    greens_frame = ttk.Frame(frm)
    greens_frame.grid(row=4, column=1, columnspan=4, sticky="w", padx=PADX, pady=PADY)
    # set columns to share space
    for c in range(5):
        greens_frame.columnconfigure(c, weight=1)
    g_vars = {}
    green_list = ['max_env','hot_spot','block_farmers','max_diff','random']
    for i, g in enumerate(green_list):
        v = tk.BooleanVar(value=(g in vals['greens']))
        g_vars[g] = v
        ttk.Checkbutton(greens_frame, text=g, variable=v).grid(row=0, column=i, sticky="w", padx=(8, 16))

    # ----------------- row 5: Farmer strategies (even spacing) --------------
    ttk.Label(frm, text="Farmer strategies").grid(row=5, column=0, sticky="w", padx=PADX, pady=PADY)
    farmers_frame = ttk.Frame(frm)
    farmers_frame.grid(row=5, column=1, columnspan=4, sticky="w", padx=PADX, pady=PADY)
    for c in range(2):
        farmers_frame.columnconfigure(c, weight=1)
    f_vars = {}
    farmer_list = ['naive','strategic']
    for i, f in enumerate(farmer_list):
        v = tk.BooleanVar(value=(f in vals['farmers']))
        f_vars[f] = v
        ttk.Checkbutton(farmers_frame, text=f, variable=v).grid(row=0, column=i, sticky="w", padx=(8, 16))

    # ----------------- row 6: Leakages ---------------------
    ttk.Label(frm, text="Leakages (comma)").grid(row=6, column=0, sticky="w", padx=PADX, pady=PADY)
    leak_var = tk.StringVar(value=vals['leakages'])
    ttk.Entry(frm, textvariable=leak_var, width=18).grid(row=6, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 7: Grid size (limit to 30) -----
    ttk.Label(frm, text="Grid size (n)").grid(row=7, column=0, sticky="w", padx=PADX, pady=PADY)
    grid_var = tk.IntVar(value=vals['grid'])
    ttk.Spinbox(frm, from_=4, to=30, increment=1, textvariable=grid_var, width=8).grid(
        row=7, column=1, sticky="w", padx=PADX, pady=PADY
    )

    # ----------------- row 8: Rounds -----------------------
    ttk.Label(frm, text='Rounds ("auto" or int)').grid(row=8, column=0, sticky="w", padx=PADX, pady=PADY)
    rounds_var = tk.StringVar(value=str(vals['rounds']))
    ttk.Entry(frm, textvariable=rounds_var, width=12).grid(row=8, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 9: Output folder ----------------
    ttk.Label(frm, text="Output folder").grid(row=9, column=0, sticky="w", padx=PADX, pady=PADY)
    outdir_var = tk.StringVar(value=vals['outdir'])
    ttk.Entry(frm, textvariable=outdir_var, width=22).grid(row=9, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 10: Mode ------------------------
    ttk.Label(frm, text="Mode").grid(row=10, column=0, sticky="w", padx=PADX, pady=PADY)
    mode_var = tk.StringVar(value=vals['mode'])
    ttk.Combobox(frm, values=['all','static','dynamic','effects'], textvariable=mode_var,
                 width=10, state="readonly").grid(row=10, column=1, sticky="w", padx=PADX, pady=PADY)

    # ----------------- row 11: Seed ------------------------
    ttk.Label(frm, text="Seed").grid(row=11, column=0, sticky="w", padx=PADX, pady=PADY)
    seed_var = tk.IntVar(value=vals['seed'])
    ttk.Spinbox(frm, from_=0, to=10_000_000, increment=1, textvariable=seed_var, width=10).grid(
        row=11, column=1, sticky="w", padx=PADX, pady=PADY
    )

    chosen = {}

    def run_and_close():
        # greens/farmers from checkboxes
        sel_greens = [g for g,v in g_vars.items() if v.get()]
        if not sel_greens:
            sel_greens = ['max_env','hot_spot','block_farmers']  # keep sane default
        sel_farmers = [f for f,v in f_vars.items() if v.get()]
        if not sel_farmers:
            sel_farmers = ['naive','strategic']

        chosen['reps']        = int(reps_var.get())
        chosen['rho']         = float(rho_var.get())
        chosen['alloc']       = alloc_var.get()
        chosen['farmer_pct']  = float(farmer_pct_var.get())
        chosen['greens']      = ','.join(sel_greens)
        chosen['farmer']      = sel_farmers
        chosen['leakages']    = leak_var.get()
        chosen['grid']        = int(grid_var.get())
        chosen['rounds']      = rounds_var.get().strip()
        chosen['outdir']      = outdir_var.get().strip()
        chosen['mode']        = mode_var.get()
        chosen['risky_rule']  = vals['risky_rule']   # keep paper default; not exposed
        chosen['seed']        = int(seed_var.get())
        root.destroy()

    ttk.Button(frm, text="Run", command=run_and_close).grid(row=12, column=0, pady=(PADY+2, PADY), padx=PADX, sticky="w")
    ttk.Button(frm, text="Cancel", command=root.destroy).grid(row=12, column=1, pady=(PADY+2, PADY), padx=PADX, sticky="w")
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
# Plotting
# --------------------------

import colorsys
from matplotlib.colors import to_rgb

def _adjust_lightness(color, factor):
    """
    Return a lighter/darker shade of a matplotlib color.
    factor > 1  => lighter
    factor < 1  => darker
    """
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Scale lightness toward 1 for factor>1 or toward 0 for factor<1
    l = max(0, min(1, 1 - (1 - l) / factor)) if factor >= 1 else max(0, min(1, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)

# Base colors per strategy (tweak if you like)
STRAT_COLORS = {
    'max_env':       '#2ca02c',  # green
    'hot_spot':      '#ff7f0e',  # orange
    'block_farmers': '#1f77b4',  # blue
    'max_diff':      '#9467bd',  # purple
    'random':        '#8c564b',  # brown
}

def plot_static_overlay(outdir: Path,
                        greens: List[str],
                        farmer_strategy: str,
                        leakages: List[float],
                        rho: float,
                        grid: int,
                        rounds: int,
                        claims_green: int,
                        claims_farmer: int,
                        reps: int,
                        seed: Optional[int],
                        multi_per_round: bool,
                        risky_rule: str):
    """
    Static overlay per farmer:
      - X axis = leakage (%)
      - Curves = Green strategies
      - One figure per outcome: Conservation, Additionality, Welfare Loss
      - Shared y-limits across strategies for fair comparison
    """
    rows = []
    for g in greens:
        for L in leakages:
            res = run_replications(
                green_strategy=g, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed, multi_per_round=multi_per_round, risky_rule=risky_rule
            )
            rows.append(dict(green=g, leakage=L,
                             cons=res['mean_cons'],
                             addl=res['mean_addl'],
                             wgap=res['mean_wgap']))
    df = pd.DataFrame(rows)
    df['leakage_pct'] = (df['leakage'] * 100).round(0).astype(int)

    def overlay(y, ylabel, fname):
        y_min, y_max = df[y].min(), df[y].max()
        pad = 0.02 * (y_max - y_min + 1e-9)
        plt.figure(figsize=(7, 5))
        for g in greens:
            sub = df[df['green'] == g].sort_values('leakage')
            base = STRAT_COLORS.get(normalize_green_strategy_name(g), 'C0')
            plt.plot(sub['leakage_pct'], sub[y], marker='o', linewidth=2,
                     color=base, label=g)
        plt.ylim(y_min - pad, y_max + pad)
        plt.xlabel('Leakage (%)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Leakage — Farmer={farmer_strategy}')
        plt.legend(title='Strategy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=220)
        plt.close()

    overlay('cons',  'Conservation',       f'static_overlay_cons_{farmer_strategy}.png')
    overlay('addl',  'Additionality',      f'static_overlay_addl_{farmer_strategy}.png')
    overlay('wgap',  'Welfare Loss (%)',   f'static_overlay_welfare_{farmer_strategy}.png')

def plot_dynamic_overlay_paper_style(outdir: Path,
                                     greens: List[str],
                                     farmer_strategy: str,
                                     leakages: List[float],
                                     rho: float,
                                     grid: int,
                                     rounds: int,
                                     claims_green: int,
                                     claims_farmer: int,
                                     reps: int,
                                     seed: Optional[int],
                                     multi_per_round: bool,
                                     risky_rule: str):
    """
    Paper-style dynamic overlay per farmer:
      - One figure per outcome (Conservation, Additionality)
      - Curves = strategies; leakage shown as lighter->darker shades per strategy
      - 100% leakage = lightest; 0% = darkest
      - Y-limits locked across strategies for fair comparison
    """
    if not leakages:
        return

    # Compute all series first (so we can lock y-limits)
    results = {}  # (g, L) -> res dict
    cons_min = cons_max = None
    add_min  = add_max  = None

    for g in greens:
        for L in leakages:
            res = run_replications(
                green_strategy=g, farmer_strategy=farmer_strategy, leakage=L,
                rho=rho, grid=grid, rounds=rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=reps, seed=seed, multi_per_round=multi_per_round, risky_rule=risky_rule
            )
            results[(g, L)] = res
            cmin, cmax = np.min(res['mean_cons_ts']), np.max(res['mean_cons_ts'])
            amin, amax = np.min(res['mean_addl_ts']), np.max(res['mean_addl_ts'])
            cons_min = cmin if cons_min is None else min(cons_min, cmin)
            cons_max = cmax if cons_max is None else max(cons_max, cmax)
            add_min  = amin if add_min  is None else min(add_min,  amin)
            add_max  = amax if add_max  is None else max(add_max,  amax)

    # Lightness mapping: 100% leakage (1.0) -> lightest; 0% -> darkest
    Ls = sorted(leakages)  # e.g., [0.0, 0.5, 1.0]
    if len(Ls) == 1:
        leak_to_factor = {Ls[0]: 1.0}
    else:
        light, dark = 1.6, 0.7  # tweak to taste
        leak_to_factor = {L: light + (dark - light) * (1.0 - L) for L in Ls}

    # --- Conservation overlay ---
    pad = 0.02 * (cons_max - cons_min + 1e-9)
    plt.figure(figsize=(7, 5))
    for g in greens:
        base = STRAT_COLORS.get(normalize_green_strategy_name(g), 'C0')
        for L in Ls:
            res = results[(g, L)]
            x = np.arange(1, len(res['mean_cons_ts']) + 1)
            color = _adjust_lightness(base, leak_to_factor[L])
            label = g if L == Ls[0] else None  # label only darkest line per strategy
            plt.plot(x, res['mean_cons_ts'], color=color, linewidth=2, label=label)
    plt.ylim(cons_min - pad, cons_max + pad)
    plt.xlabel('Round (final point includes DLE)')
    plt.ylabel('Conservation')
    plt.title(f'Dynamic Conservation — Farmer={farmer_strategy}')
    plt.legend(title='Strategy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f'dyn_overlay_conservation_{farmer_strategy}.png', dpi=220)
    plt.close()

    # --- Additionality overlay ---
    pad2 = 0.02 * (add_max - add_min + 1e-9)
    plt.figure(figsize=(7, 5))
    for g in greens:
        base = STRAT_COLORS.get(normalize_green_strategy_name(g), 'C0')
        for L in Ls:
            res = results[(g, L)]
            x = np.arange(1, len(res['mean_addl_ts']) + 1)
            color = _adjust_lightness(base, leak_to_factor[L])
            label = g if L == Ls[0] else None
            plt.plot(x, res['mean_addl_ts'], color=color, linewidth=2, label=label)
    plt.ylim(add_min - pad2, add_max + pad2)
    plt.xlabel('Round (final includes residual BAU bump if any)')
    plt.ylabel('Additionality')
    plt.title(f'Dynamic Additionality — Farmer={farmer_strategy}')
    plt.legend(title='Strategy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f'dyn_overlay_additionality_{farmer_strategy}.png', dpi=220)
    plt.close()


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Conservation Strategy Game — Replication-Ready Simulator (v2)")
    ap.add_argument('--mode', choices=['static','dynamic','effects','all'], default='all',
                    help='Which outputs to produce.')
    ap.add_argument('--farmer', choices=['naive', 'strategic'], nargs='+',
                    default=['naive', 'strategic'],
                    help='One or more farmer strategies to run (default: both).')
    ap.add_argument('--greens', default='max_env,hot_spot,block_farmers',
                    help='Comma-separated green strategies: max_env,hot_spot,block_farmers,max_diff,max_difference,random')
    ap.add_argument('--leakages', default='1.0,0.5,0.0',
                    help='Comma-separated leakage values in [0,1], e.g., 1.0,0.5,0.0')
    ap.add_argument('--rho', type=float, default=0.0, help='Correlation between E and A')
    ap.add_argument('--grid', type=int, default=10, help='Grid size n (n x n cells)')
    ap.add_argument('--rounds', type=int, default=50, help='Max rounds (Farmer then Green per round)')
    ap.add_argument('--reps', type=int, default=500, help='Monte Carlo replications')
    ap.add_argument('--alloc', choices=['equal','political'], default='equal', help='Claims allocation rule')
    ap.add_argument('--farmer_pct', type=float, default=0.7, help='Farmer share (political alloc)')
    ap.add_argument('--risky-rule', choices=['green_claims','farmer_claims'], default='green_claims',
                    help='Strategic Farmer risky set size keyed to Green or Farmer claims.')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--outdir', default='outputs', help='Directory for figures/tables')
    return ap.parse_args()


def main():
    args = parse_args()  # keep CLI parsing for super users / automation

    # --- UI is default: show menu if available; else fall back to CLI args ---
    if USE_UI_DEFAULT and _gui_available():
        ui = launch_ui_and_get_args()
        if ui:
            # overwrite CLI args with UI selections
            args.reps = ui['reps']
            args.rho = ui['rho']
            args.alloc = ui['alloc']
            args.farmer_pct = ui['farmer_pct']
            args.greens = ui['greens']
            args.farmer = ui['farmer']
            args.leakages = ui['leakages']
            args.grid = ui['grid']
            args.rounds = ui['rounds']
            args.outdir = ui['outdir']
            args.mode = ui['mode']
            args.risky_rule = ui['risky_rule']
            args.seed = ui['seed']
        else:
            print("[info] UI cancelled or returned no values; using CLI arguments.")
    else:
        if USE_UI_DEFAULT:
            print("[info] UI not available in this environment; using CLI arguments.")

    # --- Defensive clip of rho ---
    if args.rho < -1.0 or args.rho > 1.0:
        print(f"[warn] rho={args.rho} out of range [-1,1]; clipping.")
        args.rho = max(-1.0, min(1.0, args.rho))

    outdir = ensure_outdir(args.outdir)

    # --- Parse strategy & leakage lists ---
    greens = [normalize_green_strategy_name(s) for s in args.greens.split(',') if s.strip()]
    leakages = sorted(float(x) for x in args.leakages.split(',') if x.strip())

    # --- Claims from grid & allocation ---
    total_cells = args.grid * args.grid
    claims_green, claims_farmer = allocate_points(
        total_cells, allocation=args.alloc, farmer_percentage=args.farmer_pct
    )

    # --- Rounds (auto => one claim per side per round); else integer) ---
    if isinstance(args.rounds, str) and args.rounds.strip().lower() == 'auto':
        computed_rounds = max(claims_green, claims_farmer)  # default: no batching
    else:
        try:
            computed_rounds = int(args.rounds)
            if computed_rounds <= 0:
                print(f"[warn] rounds={args.rounds} invalid; using auto.")
                computed_rounds = max(claims_green, claims_farmer)
        except Exception:
            print(f"[warn] rounds={args.rounds} invalid; using auto.")
            computed_rounds = max(claims_green, claims_farmer)

    # --- Batching derives solely from rounds ---
    batching = (computed_rounds < max(claims_green, claims_farmer))

    # --- Info log ---
    print(f"[info] farmer={args.farmer}, greens={greens}, leakages={leakages}, "
          f"grid={args.grid}, claims=(G={claims_green}, F={claims_farmer}), "
          f"rounds={computed_rounds}, batching={batching}, rho={args.rho}, reps={args.reps}")

    # --- Effects + Plots ---
    for farmer_strategy in args.farmer:

        # Effects table
        if args.mode in ('effects', 'all'):
            df = compute_effects_table(
                greens=greens,
                farmer_strategy=farmer_strategy,
                leakages=leakages,
                rho=args.rho,
                grid=args.grid,
                rounds=computed_rounds,
                claims_green=claims_green,
                claims_farmer=claims_farmer,
                reps=args.reps,
                seed=args.seed,
                multi_per_round=batching,
                risky_rule=args.risky_rule
            )
            csv_path = outdir / f'effects_table_{farmer_strategy}.csv'
            df.to_csv(csv_path, index=False)
            print(f"[saved] {csv_path}")

        # Static overlays (leakage on X; curves = strategies)
        if args.mode in ('static', 'all'):
            plot_static_overlay(
                outdir=outdir,
                greens=greens,
                farmer_strategy=farmer_strategy,
                leakages=leakages,
                rho=args.rho, grid=args.grid, rounds=computed_rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=args.reps, seed=args.seed,
                multi_per_round=batching, risky_rule=args.risky_rule
            )
            print(f"[saved] static overlay plots for farmer={farmer_strategy}")

        # Dynamic overlays (curves = strategies; leakage as lighter→darker shades)
        if args.mode in ('dynamic', 'all'):
            plot_dynamic_overlay_paper_style(
                outdir=outdir,
                greens=greens,
                farmer_strategy=farmer_strategy,
                leakages=leakages,
                rho=args.rho, grid=args.grid, rounds=computed_rounds,
                claims_green=claims_green, claims_farmer=claims_farmer,
                reps=args.reps, seed=args.seed,
                multi_per_round=batching, risky_rule=args.risky_rule
            )
            print(f"[saved] dynamic overlay (paper-style) for farmer={farmer_strategy}")

    print("[done]")


if __name__ == '__main__':
    main()
