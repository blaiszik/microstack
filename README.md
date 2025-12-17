# ATOMICS - AI Materials Scientist

A CLI agent for analyzing atomic surfaces using Machine Learning Potentials, with experimental validation and AI-generated scientific reports.

## Features

- **Surface Generation**: Create atomic surfaces for FCC metals (Cu, Pt, Au, Ag, Ni, Pd) and 2D materials (Graphene, MoS2)
- **ML Relaxation**: Relax surfaces using MACE-MP potential trained on Materials Project DFT data
- **Experimental Validation**: Compare predictions against LEED/DFT reference data from literature
- **AI Scientific Reports**: Generate publication-style reports with Claude-powered discussion sections
- **Rich CLI**: Interactive terminal with markdown rendering

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mic-hack
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (in `config.py` or environment variables):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."  # For AI discussion generation
   export MP_API_KEY="..."                 # For Materials Project queries (optional)
   ```

## Usage

```bash
python agent_example.py
```

### Commands

| Command | Description |
|---------|-------------|
| `analyze Cu 100` | Full analysis pipeline with AI report |
| `analyze Pt 111` | Analyze platinum (111) surface |
| `list references` | Show available experimental data |
| `relax Ni 100` | Quick relaxation without report |
| `generate graphene` | Just create the structure |

### Example Session

```
User> analyze Cu 100

[1/5] Generating surface structure...
      Created Cu(100) with 36 atoms
[2/5] Loading MACE model...
[3/5] Relaxing surface...
      Energy: -126.4521 → -126.8934 eV (Δ = -0.4413 eV)
[4/5] Generating visualization...
[5/5] Analyzing and generating report...
      Comparing with: Lindgren et al., Phys. Rev. B 29, 576 (1984)
      Agreement: GOOD

# Surface Relaxation Analysis: Cu(100)
...
```

## How It Works

### The Analysis Pipeline

When you run `analyze Cu 100`, the following steps occur:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ANALYSIS PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: SURFACE GENERATION
├── Input: Element (Cu) + Face (100)
├── Process: ASE builds FCC lattice with experimental lattice constant
├── Output: 3×3×4 supercell (36 atoms) with 10Å vacuum
└── File: Cu_100_unrelaxed.xyz

                              ↓

Step 2: MODEL LOADING
├── Load MACE-MP-0 medium model
├── Pre-trained on ~150k Materials Project DFT calculations
└── Predicts energies and forces for any element combination

                              ↓

Step 3: SURFACE RELAXATION
├── Algorithm: FIRE (Fast Inertial Relaxation Engine)
├── Process: Iteratively move atoms to minimize energy
├── Steps: 200 optimization steps
├── Physics: Surface atoms relax inward (Smoluchowski smoothing)
└── File: Cu_100_relaxed.xyz

                              ↓

Step 4: VISUALIZATION
├── Top panel: Side view colored by z-displacement
│   └── Blue = contracted, Red = expanded
├── Bottom panel: Interlayer spacing changes (%)
└── File: Cu_100_relaxation.png

                              ↓

Step 5: ANALYSIS & COMPARISON
├── Extract metrics:
│   ├── d₁₂ change (top layer spacing)
│   ├── d₂₃ change (second layer spacing)
│   └── Atomic displacements
├── Query reference data:
│   ├── Materials Project (bulk properties)
│   └── Literature LEED data (surface relaxations)
└── Compute agreement score

                              ↓

Step 6: AI REPORT GENERATION
├── Send structured data to Claude API
├── Claude writes scientific discussion:
│   ├── Physical interpretation (why does it relax?)
│   ├── Comparison analysis (how accurate is MACE?)
│   └── Implications (catalysis, microscopy applications)
└── File: Cu_100_report.md
```

### What Each Module Does

| Module | Purpose |
|--------|---------|
| `generate_surfaces.py` | Creates atomic structures using ASE's surface builders |
| `surface_relaxation.py` | Loads MACE model and runs FIRE optimization via TorchSim |
| `materials_project.py` | Queries MP API + stores curated LEED reference data |
| `comparison.py` | Extracts relaxation metrics and computes agreement scores |
| `report_generator.py` | Assembles markdown report and calls Claude for discussion |
| `agent_example.py` | LangGraph agent with CLI interface |
| `config.py` | API keys and settings |

### The Science Behind Surface Relaxation

**Why do surfaces relax?**

Surface atoms have fewer neighbors than bulk atoms. This creates an imbalance:

```
Bulk atom:     12 nearest neighbors (FCC)
Surface atom:  8-9 nearest neighbors (missing atoms above)
```

The reduced coordination causes:
1. **Smoluchowski smoothing**: Electron density redistributes, pulling surface atoms inward
2. **d₁₂ contraction**: Top layer moves toward second layer (typically -1% to -3%)
3. **d₂₃ expansion**: Second layer compensates with slight outward movement
4. **Oscillatory damping**: Effect decays into the bulk

**Reference Data Sources**

| Method | What it measures |
|--------|------------------|
| LEED (Low-Energy Electron Diffraction) | Atomic positions via electron scattering |
| DFT (Density Functional Theory) | Quantum mechanical energy minimization |
| MACE (This project) | ML potential trained on DFT data |

## Available Reference Data

| Element | Surfaces | Reference |
|---------|----------|-----------|
| Cu | 100, 111, 110 | Lindgren (1984), Davis (1983), Adams (1987) |
| Pt | 100, 111 | Heilmann (1979), Materer (1995) |
| Au | 100, 111 | Gibbs (1991), Harten (1985) |
| Ag | 100, 111 | Quinn (1988), Soares (1999) |
| Ni | 100, 111 | Demuth (1975), Narasimhan (1992) |
| Pd | 100, 111 | Behm (1983), Ohtani (1987) |
| C | graphene | Castro Neto (2009) |
| MoS2 | 2d | Splendiani (2010) |

## Output Files

| File | Contents |
|------|----------|
| `{element}_{face}_unrelaxed.xyz` | Initial atomic structure (XYZ format) |
| `{element}_{face}_relaxed.xyz` | Relaxed atomic structure |
| `{element}_{face}_relaxation.png` | Visualization of relaxation |
| `{element}_{face}_report.md` | Full scientific report with AI discussion |

## Project Structure

```
mic-hack/
├── agent_example.py      # Main CLI agent
├── generate_surfaces.py  # Surface generation (ASE)
├── surface_relaxation.py # ML relaxation (MACE + TorchSim)
├── materials_project.py  # Reference data + MP API
├── comparison.py         # Analysis engine
├── report_generator.py   # Report + Claude integration
├── config.py             # API keys and settings
└── requirements.txt      # Dependencies
```

## License

MIT License
