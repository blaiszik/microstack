"""Surface Relaxation for Microscopy with TorchSim.

This example demonstrates batched surface relaxation using the MACE model.
It creates Cu(100) and Pt(111) surfaces using ASE and relaxes them simultaneously
using the FIRE optimizer, showcasing TorchSim's batching capabilities for
surface preparation relevant to microscopy imaging.

To use Fairchem instead, see the commented code in the model loading section.
Note: MACE and Fairchem have conflicting dependencies and cannot be installed together.
"""

# /// script
# dependencies = ["ase>=3.26", "mace-torch>=0.3.12", "matplotlib"]
# ///

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ase import Atoms
from ase.build import fcc100, fcc111
from ase.io import write
from mace.calculators.foundations_models import mace_mp
import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

SMOKE_TEST = os.getenv("CI") is not None
N_STEPS_DEFAULT = 10 if SMOKE_TEST else 200
ELEMENT_COLORS = {"Cu": "#B87333", "Pt": "#E5E4E2"}


# =============================================================================
# Visualization Function
# =============================================================================


def plot_surface_relaxation(
    unrelaxed: list[Atoms],
    relaxed: list[Atoms],
    titles: list[str],
    filename: str = "surface_relaxation.png",
) -> None:
    """Visualize surface relaxation showing structure and atomic displacements.

    Top row: Side view of relaxed structure with atoms colored by z-displacement
    Bottom row: Layer-by-layer analysis of interlayer spacing changes
    """
    n_surfaces = len(unrelaxed)
    fig, axes = plt.subplots(2, n_surfaces, figsize=(6 * n_surfaces, 10))

    # Handle single surface case
    if n_surfaces == 1:
        axes = axes.reshape(2, 1)
        # If axes became 2D but we iterate, we need to handle it carefully.
        # But below loop uses zip(unrelaxed...) so it iterates n_surfaces times.
        # We need to access axes correctly.
        # If n_surfaces=1, axes is (2, 1). axes[0, 0] is top, axes[1, 0] is bottom.
    
    # If n_surfaces > 1, axes is (2, n). axes[0, i] is top, axes[1, i] is bottom.

    for i, (before, after, title) in enumerate(zip(unrelaxed, relaxed, titles)):
        element = before.get_chemical_symbols()[0]
        base_color = ELEMENT_COLORS.get(element, "#808080")

        pos_before = before.get_positions()
        pos_after = after.get_positions()

        # Calculate z-displacement for each atom
        z_displacement = pos_after[:, 2] - pos_before[:, 2]

        # Top panel: Side view with atoms colored by displacement
        if n_surfaces == 1:
            ax = axes[0, 0]
        else:
            ax = axes[0, i]

        # Color atoms by z-displacement (blue=down, red=up)
        max_disp = max(abs(z_displacement.min()), abs(z_displacement.max()), 0.01)
        
        scatter = ax.scatter(
            pos_after[:, 0],
            pos_after[:, 2],
            c=z_displacement,
            cmap="coolwarm",
            vmin=-max_disp,
            vmax=max_disp,
            s=150,
            edgecolors="black",
            linewidth=0.5,
        )
        cbar = plt.colorbar(scatter, ax=ax, label="z-displacement (A)")

        ax.set_xlabel("x (A)")
        ax.set_ylabel("z (A)")
        ax.set_title(f"{title}\nSide View (colored by z-displacement)")
        ax.axhline(
            y=pos_after[:, 2].max() + 2, color="gray", linestyle="--", alpha=0.5
        )
        ax.text(
            pos_after[:, 0].mean(),
            pos_after[:, 2].max() + 3.5,
            "vacuum",
            ha="center",
            color="gray",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

        # Bottom panel: Layer-by-layer interlayer spacing analysis
        if n_surfaces == 1:
            ax = axes[1, 0]
        else:
            ax = axes[1, i]

        # Identify layers by clustering z-coordinates
        z_before = pos_before[:, 2]
        z_after = pos_after[:, 2]

        # Get unique layer z-positions (rounded to identify layers)
        layer_z_before = np.unique(np.round(z_before, decimals=1))
        layer_z_after = []

        # Calculate mean z for each layer after relaxation
        for z_layer in layer_z_before:
            mask = np.abs(z_before - z_layer) < 0.5
            layer_z_after.append(np.mean(z_after[mask]))

        layer_z_after = np.array(layer_z_after)

        # Calculate interlayer spacings
        spacings_before = np.diff(layer_z_before)
        spacings_after = np.diff(layer_z_after)
        
        if len(spacings_before) > 0:
            spacing_change_pct = (spacings_after - spacings_before) / spacings_before * 100

            # Plot interlayer spacing changes
            layer_labels = [f"d{j+1}{j+2}" for j in range(len(spacings_before))]
            x_pos = np.arange(len(layer_labels))

            bars = ax.bar(x_pos, spacing_change_pct, color=base_color, edgecolor="black")

            # Color bars by sign (contraction=blue, expansion=red)
            for bar, change in zip(bars, spacing_change_pct):
                bar.set_color("steelblue" if change < 0 else "indianred")

            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_xlabel("Interlayer spacing")
            ax.set_ylabel("Change (%)")
            ax.set_title(f"Interlayer Spacing Relaxation\n(negative = contraction)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layer_labels)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, change in zip(bars, spacing_change_pct):
                height = bar.get_height()
                ax.annotate(
                    f"{change:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=9,
                )
        else:
            ax.text(0.5, 0.5, "Not enough layers for spacing analysis", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {filename}")


# =============================================================================
# Model Loading (MACE)
# =============================================================================

def load_model(device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype=str(dtype).removeprefix("torch."),
        device=str(device),
    )
    model = MaceModel(
        model=loaded_model,
        device=device,
        compute_forces=True,
        compute_stress=False,  # Stress not needed for surface optimization
        dtype=dtype,
        enable_cueq=False,
    )
    print("Using MACE model")
    return model

# =============================================================================
# Batched Relaxation
# =============================================================================

def relax_surfaces(surfaces: list[Atoms], model, steps=N_STEPS_DEFAULT, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert ASE atoms to TorchSim state (batched)
    state = ts.io.atoms_to_state(surfaces, device=device, dtype=dtype)
    print(f"  TorchSim PBC: {state.pbc}")

    # Run initial inference to get starting energies
    results = model(state)
    initial_energies = results["energy"].tolist()
    
    # Initialize FIRE optimizer (no cell filter - fixed cell for surfaces)
    state = ts.fire_init(state=state, model=model, dt_start=0.005)

    # Run optimization
    print(f"\nRunning FIRE optimization ({steps} steps):")
    for step in range(steps):
        if step % 20 == 0:
            energies = state.energy.tolist()
            # print(f"  Step {step:4d}, Energies: {energies}")

        state = ts.fire_step(state=state, model=model, dt_max=0.01)

    final_energies = state.energy.tolist()
    
    # Convert final state back to ASE atoms
    relaxed_surfaces = ts.io.state_to_atoms(state)
    
    return relaxed_surfaces, initial_energies, final_energies


def main():
    # Device and dtype setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    model = load_model(device, dtype)

    # =============================================================================
    # Surface Creation with ASE
    # =============================================================================

    # Cu(100) surface: 3x3 unit cells, 4 atomic layers, 10 Angstrom vacuum
    # Using experimental lattice constant a=3.6 Angstrom for Cu
    cu_surface = fcc100("Cu", size=(3, 3, 4), a=3.6, vacuum=10.0)

    # Pt(111) surface: 3x3 unit cells, 4 atomic layers, 10 Angstrom vacuum
    # Using experimental lattice constant a=3.92 Angstrom for Pt
    pt_surface = fcc111("Pt", size=(3, 3, 4), a=3.92, vacuum=10.0)

    surfaces = [cu_surface, pt_surface]
    surface_names = ["Cu(100)", "Pt(111)"]

    # Store copies of unrelaxed structures for visualization
    unrelaxed_surfaces = [atoms.copy() for atoms in surfaces]

    print(f"\nSurface structures created:")
    print(f"  Cu(100): {len(cu_surface)} atoms, PBC: {cu_surface.pbc}")
    print(f"  Pt(111): {len(pt_surface)} atoms, PBC: {pt_surface.pbc}")
    
    relaxed_surfaces, initial_energies, final_energies = relax_surfaces(surfaces, model, steps=N_STEPS_DEFAULT, device=device, dtype=dtype)

    print(f"\nResults:")
    print(f"  Cu(100): {initial_energies[0]:.4f} -> {final_energies[0]:.4f} eV "
          f"(change: {final_energies[0] - initial_energies[0]:.4f} eV)")
    print(f"  Pt(111): {initial_energies[1]:.4f} -> {final_energies[1]:.4f} eV "
          f"(change: {final_energies[1] - initial_energies[1]:.4f} eV)")

    # Save relaxed structures to disk
    for atoms, name in zip(relaxed_surfaces, surface_names):
        filename = f"{name.lower().replace('(', '_').replace(')', '')}_relaxed.xyz"
        write(filename, atoms)
        print(f"  Saved: {filename}")

    # =============================================================================
    # Visualization
    # =============================================================================

    plot_surface_relaxation(
        unrelaxed=unrelaxed_surfaces,
        relaxed=relaxed_surfaces,
        titles=surface_names,
        filename="surface_relaxation.png",
    )

if __name__ == "__main__":
    main()

