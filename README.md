# Adversarial Procurement (Conservation Strategy) — Replication-Ready Simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17114490.svg)](https://doi.org/10.5281/zenodo.17114490)

Documentation:
[User Manual (PDF)]:  adversarial_procurement_replication_user_guide.pdf

This repository contains the **Adversarial Procurement Replication**, a Monte Carlo simulator of contested conservation strategies ("Greens" vs "Farmers"). The simulator produces:

* **Static overlay plots** (final outcomes vs leakage)
* **Dynamic overlay plots** (time paths of conservation and additionality)
* **Effects tables (CSV)** decomposing outcomes into Pure Strategy Effect (PSE) and Displacement–Leakage Effect (DLE)

The simulator replicates the analyses and figures from the working paper:

> Weinhold, D. & Andersen, L. E. (2026).
> *Adversarial Procurement in Two-Value Space: Insights and Evidence for Conservation Siting*
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
python Adversarial_Procurement_Replication.py   # launches the popup UI
```

### Command-line usage

For advanced users, the simulator can run without the UI:

```bash
# Claims World, strategic farmers only:
python Adversarial_Procurement_Replication.py --world claims --farmer strategic

# Budget World with a 70--30 political split in favour of Farmers:
python Adversarial_Procurement_Replication.py --world budget --alloc political --farmer\_pct 0.7

# Claims World with a custom strategy subset and leakage grid:
python Adversarial_Procurement_Replication.py --world claims --greens max\_env,hot\_spot,block\_farmers --leakages 1.0,0.5,0.0

# Budget World with advanced options:
python Adversarial_Procurement_Replication.py --world budget --budget\_theta 0.0 --hotspot\_additive off
```

Options include:

* `--greens`: green strategies (max\_env, hot\_spot, block\_farmers, max\_diff, random)
* `--alloc`: allocation rule (equal or political)
* `--farmer_pct`: farmer share if allocation = political
* `--rho`: correlation between environmental and agricultural values
* `--reps`: number of Monte Carlo replications
* `--outdir`: output directory (default `outputs/`)
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

> Weinhold, D. & Andersen, L. E. (2026).
> *Adversarial Procurement in Two-Value Space: \\ Insights and Evidence for Conservation Siting*
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
