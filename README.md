# Surface Agent CLI

A powerful CLI agent for generating and relaxing atomic surfaces using Machine Learning Potentials (MACE).

## Features

- **Surface Generation**: Create atomic surfaces for various elements (Cu, Pt, Au, Ag, Al, Ni, Pd, Fe, Ir, Rh).
- **2D Materials**: Support for Graphene and Transition Metal Dichalcogenides (e.g., MoS2, WS2).
- **Surface Relaxation**: Relax surfaces using the MACE model and FIRE optimizer.
- **Visualization**: Automatically generates plots showing atomic displacements and interlayer spacing changes.
- **Interactive CLI**: Easy-to-use command-line interface powered by Langgraph.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mic-hack
   ```

2. **Install dependencies**:
   ```bash
   pip install ase mace-torch matplotlib langgraph langchain-core
   ```
   *Note: You may need to install `torch` separately depending on your system (CUDA/CPU).*

## Usage

Run the agent:

```bash
python agent_example.py
```

### Commands

Once inside the CLI, you can use natural language commands:

- **Generate a surface**:
  - `generate Cu 100`
  - `create Pt 111`
  - `generate graphene`

- **Relax a surface**:
  - `relax Au 110`
  - `relax MoS2 2d`
  - `relax C graphene`

### Output

The agent will produce:
- `*_unrelaxed.xyz`: The initial atomic structure.
- `*_relaxed.xyz`: The relaxed atomic structure.
- `*_relaxation.png`: A visualization of the relaxation process.

## Structure

- `agent_example.py`: Main entry point containing the Langgraph agent and CLI logic.
- `generate_surfaces.py`: Logic for creating atomic structures using ASE.
- `surface_relaxation.py`: Core logic for loading the MACE model and running relaxations.

## License

[License information here]