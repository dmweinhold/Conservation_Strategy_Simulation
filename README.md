# Conservation Strategy Game — Replication-Ready Simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17114490.svg)](https://doi.org/10.5281/zenodo.17114490)

This repository contains the **Conservation Strategy Game**, a Monte Carlo simulator of contested conservation strategies ("Greens" vs "Farmers"). The simulator produces:

* **Static overlay plots** (final outcomes vs leakage)
* **Dynamic overlay plots** (time paths of conservation and additionality)
* **Effects tables (CSV)** decomposing outcomes into Pure Strategy Effect (PSE) and Displacement–Leakage Effect (DLE)

The simulator replicates the analyses and figures from the working paper:

> Weinhold, D. & Andersen, L. E. (2025).
> *Conservation Strategies in Contested Environments: Dynamic Monte Carlo Simulations and a Bolivian Case Study.*
> London School of Economics & Universidad Privada Boliviana. Working Paper.
> [DOI: 10.5281/zenodo.17114490](https://doi.org/10.5281/zenodo.17114490)

---

## Getting Started

### Requirements

* Python 3.10+
* Packages listed in `requirements.txt`:

  * numpy
  * pandas
  * matplotlib
  * pyyaml

### Quick start

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
python conservation_game_sim.py   # launches the popup UI
```

### Command-line usage

For advanced users, the simulator can run without the UI:

```bash
# Run all outputs for both farmer strategies
python conservation_game_sim.py --mode all --farmer naive --farmer strategic

# Static outcomes for strategic farmers only
python conservation_game_sim.py --mode static --farmer strategic --leakages 1.0,0.5,0.0
```

Options include:

* `--greens`: green strategies (max\_env, hot\_spot, block\_farmers, max\_diff, random)
* `--alloc`: allocation rule (equal or political)
* `--farmer_pct`: farmer share if allocation = political
* `--rho`: correlation between environmental and agricultural values
* `--rounds`: number of rounds (`auto` = one claim per side per round)
* `--reps`: number of Monte Carlo replications
* `--outdir`: output directory (default `outputs/`)
* `--risky-rule`: strategic farmer risky set (`green_claims` or `farmer_claims`)
* `--seed`: random seed

---

## Outputs

Results are saved into the `outputs/` folder:

* **Overlay plots** (`*.png`)
* **Effects tables** (`effects_table_*.csv`)

By default, the repo ignores generated PNG/CSV files via `.gitignore` (only a placeholder `.gitkeep` is tracked).

---

## Citation

If you use this simulator in teaching, research, or publications, please cite:

> Weinhold, D. & Andersen, L. E. (2025).
> *Conservation Strategies in Contested Environments: Dynamic Monte Carlo Simulations and a Bolivian Case Study.*
> London School of Economics & Universidad Privada Boliviana. Working Paper.
> DOI: [10.5281/zenodo.17114490](https://doi.org/10.5281/zenodo.17114490)

You can also use GitHub’s **“Cite this repository”** button for BibTeX, APA, and other formats.

---

## License

**All rights reserved.**
See the `LICENSE` file for details.

---

## Contact

Author: **Diana Weinhold**
Email: [d.weinhold@lse.ac.uk](mailto:d.weinhold@lse.ac.uk)
GitHub: [dmweinhold](https://github.com/dmweinhold)
Repository: [Conservation\_Strategy\_Simulation](https://github.com/dmweinhold/Conservation_Strategy_Simulation)
