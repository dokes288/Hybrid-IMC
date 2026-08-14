# Hybrid-IMC

**Hybrid In-Memory Computing Design Based on Data Movement for Transformer Inference Acceleration**

A Python-based simulation and benchmarking framework comparing digital CMOS, ReRAM analog in-memory computing (AIMC), FeFET compute-in-memory (CIM), and hybrid SRAM+ReRAM architectures for BERT-Base transformer inference. This repository supports my PhD dissertation research at Prairie View A&M University and the methodology behind an IEEE manuscript currently under review: *"Transcending Von Neumann: A Comparative Analysis of In-Memory AI Accelerators for Energy-Efficient Transformer Model Inference"* (IEEE Open Journal of the Computer Society, under review).

---

## Motivation

Transformer-based models are increasingly bottlenecked by data movement rather than raw compute — the classic von Neumann bottleneck. In-memory computing (IMC) addresses this by performing computation directly within or near memory, but different IMC implementations (analog, digital, hybrid) make very different tradeoffs in precision, area, and energy efficiency.

This project builds a **seven-layer validated benchmarking methodology** to compare these tradeoffs on a shared, controlled-variable basis, rather than relying on isolated results scattered across the literature that use different workloads, precisions, or simulation frameworks.

---

## The Four Configurations Compared

| Configuration | Compute Paradigm | Technology Node |
|---|---|---|
| **Baseline** | Digital CMOS systolic MAC (TPU v4 MXU-equivalent) | 7 nm |
| **Config A** | ReRAM Analog In-Memory Computing (AIMC), 1T1R HfO₂ | 22 nm |
| **Config B** | FeFET Compute-in-Memory (threshold-voltage-domain MAC) | 28 nm |
| **Config C** | Hybrid: SRAM CIM (attention) + ReRAM AIMC (feedforward) | 5 nm / 22 nm |

**Workload:** BERT-Base-uncased, fine-tuned on GLUE SST-2, sequence length 128.

---

## The Seven-Layer Validation Methodology

Every configuration is evaluated against the same seven validation criteria to ensure fair, controlled-variable comparison:

1. **Shared baseline** — identical SRAM buffer hierarchy, ADC/DAC parameters, packaging overhead, and DRAM interface across all four configs
2. **Common simulation framework** — all configs modeled through the same NeuroSim V1.4-based analytical pipeline
3. **1b-normalized TOPS/W** — bit-normalized efficiency metric enabling fair comparison across different weight precisions
4. **Iso-accuracy targeting** — each configuration operates at the precision required to hit a 90% SST-2 accuracy target, not an arbitrary fixed precision
5. **Array / chip / system level reporting** — results reported at all three levels of abstraction, not just idealized array-level numbers
6. **Technology-node normalization** — all configs normalized to a common 7nm reference using IRDS 2022 scaling factors, so results aren't confounded by process node differences
7. **Full parameter disclosure** — every technology-intrinsic parameter (Ron/Roff ratios, Vth window, endurance, etc.) is disclosed for reproducibility

---

## Key Findings

Based on the full-system simulation results in this repository:

- **DRAM access dominates system-level energy** across all configurations — 49-53% of total system energy in most configs, confirming data movement (not compute) as the primary bottleneck
- **Config C (Hybrid SRAM CIM + ReRAM)** achieves the best system-level energy efficiency among CIM approaches, at 2.32 system TOPS/W
- **Config A (ReRAM AIMC)** achieves 2.33 system TOPS/W with the simplest single-technology implementation
- **Config B (FeFET CIM)** trades efficiency for higher I/O overhead (71% of energy in I/O) due to its lower native throughput requiring more I/O cycles per inference
- The **TOMAS operation taxonomy** shows 97.3% of BERT-Base MAC operations (QKV projections, output projections, feedforward layers) map to static-weight AIMC-friendly execution, while only 2.7% (attention score and attention×value operations) require dynamic SRAM CIM execution — this asymmetry is the basis for the hybrid architecture's advantage

Full per-configuration results, including energy/area/latency breakdowns and sensitivity analysis (ADC resolution, DRAM technology, subarray size sweeps), are in `results/`.

---

## Repository Structure

```
Hybrid-IMC/
├── notebooks/
│   └── IMC_BERT_Survey.ipynb      # Complete simulation pipeline (Phases 1-4 + sensitivity analysis)
├── results/
│   ├── data/                       # Raw JSON results (architecture, ops mapping, per-config metrics)
│   ├── figures/                    # Publication figures (accuracy/bitwidth, system dashboard, TOMAS taxonomy)
│   ├── tables/                     # LaTeX tables ready for paper insertion
│   └── sensitivity/                # ADC, DRAM, and subarray-size sensitivity sweep results
└── README.md
```

---

## The Simulation Pipeline (`notebooks/IMC_BERT_Survey.ipynb`)

The notebook is organized into four phases, designed to run sequentially in Google Colab:

**Phase 1 — BERT Architecture & Workload Characterization**
Downloads BERT-Base-uncased, extracts architecture parameters, maps every operation to its TOMAS execution category (AIMC vs. SRAM CIM vs. digital peripheral), runs FP32 baseline inference on SST-2, and performs a quantization sweep (2-bit through 8-bit) to establish the accuracy-vs-precision curve that determines the target precision for each hardware configuration.

**Phase 2 — Per-Configuration Chip-Level Simulation**
For each of the four configurations, estimates array-level and chip-level area, latency, and energy using technology-calibrated parameters sourced from peer-reviewed silicon measurements (ISSCC, IEDM, Nature Communications — see `docs/` for full parameter provenance).

**Phase 3 — Full-System Integration**
Combines chip-level compute results with a shared memory hierarchy model (SRAM buffer via CACTI-style modeling, HBM2E/HBM3E DRAM) to produce full-system energy, latency, and throughput estimates — this is where the DRAM-dominated energy breakdown emerges.

**Phase 4 — Cross-Configuration Analysis & Publication Outputs**
Consolidates all four configurations into unified comparison tables and figures, validates results against published silicon data points (IBM, Mythic, TSMC, and other test-chip publications), and generates the LaTeX tables and figures used directly in the associated paper.

**Sensitivity Analysis**
Additional cells sweep ADC resolution, DRAM technology choice, and subarray size to characterize how sensitive the system-level results are to these design parameters.

---

## Data Provenance

Every technology parameter used in this framework (ReRAM Ron/Roff ratios, FeFET threshold voltage window, SRAM bitcell area, ADC energy-per-conversion, HBM bandwidth/energy) is sourced from a specific peer-reviewed publication — primarily ISSCC, IEDM, VLSI Symposium, and Nature Communications papers from 2020-2024. This is a deliberate design choice: rather than using idealized or vendor-marketing numbers, every parameter is traceable to a measured silicon result. See `results/data/master_config.json` for the full parameter set and inline source citations.

---

## Related Publication

D. Okeke, I. Nzekwe, S. Cui, S. M. Musa, C. M. Akujuobi, and J. Foreman, **"Transcending Von Neumann: A Comparative Analysis of In-Memory Computing for Transformer Inference,"** *IEEE Open Journal of the Computer Society* (under review), 2026.

---

## Getting Started

The primary pipeline is designed to run in Google Colab (see notebook for GPU/environment setup instructions):

```bash
# Clone the repository
git clone https://github.com/dokes288/Hybrid-IMC.git
cd Hybrid-IMC

# Open notebooks/IMC_BERT_Survey.ipynb in Google Colab
# Follow the setup cells at the top of the notebook (package installation,
# Google Drive mounting, environment verification)
```

**Note on large files:** The raw BERT-Base weight checkpoint (~420MB) is not included in this repository, since it is trivially reproducible via `transformers.AutoModel.from_pretrained('bert-base-uncased')` and exceeds GitHub's practical file size guidance. The notebook downloads it automatically in Phase 1.

---

## Author

**Dominic Nze Okeke**
PhD Candidate, Electrical Engineering — Prairie View A&M University
Signal Integrity & Customer Enabling Engineer Intern, Intel Corporation (2024–2026)
ORCID: [0009-0007-6335-2628](https://orcid.org/0009-0007-6335-2628) · [LinkedIn](https://www.linkedin.com/in/dominic-okeke-601b8930/) · dnokeke@gmail.com

---

## License

*(Consider adding a license — MIT is a common, permissive choice for academic/portfolio code. Add a `LICENSE` file at the repo root once decided.)*
