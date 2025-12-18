"""Pydantic models for LLM query parsing."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    """Structured representation of a parsed microscopy query."""

    # Task type and microscopy
    task_type: Optional[
        Literal["Microscopy_Simulation", "SciLink_Structure_Generation"]
    ] = Field(
        default=None,
        description="Type of task (microscopy simulation or structure generation)",
    )
    microscopy_type: Optional[Literal["AFM", "STM", "IETS"]] = Field(
        default=None, description="Type of microscopy simulation requested"
    )

    # Material specifications
    material_formula: Optional[str] = Field(
        default=None,
        description="Chemical formula or material identifier (e.g., 'Si', 'NaCl', 'graphene')",
    )
    material_id: Optional[str] = Field(
        default=None,
        description="Specific material database ID (e.g., 'mp-149', 'oqmd-12345')",
    )
    structure_uuid: Optional[str] = Field(
        default=None,
        description="UUID of an existing generated structure",
    )
    structure_source: Optional[Literal["materials_project", "oqmd", "local_file"]] = (
        Field(
            default=None,
            description="Source for structure data",
        )
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to local structure file (XYZ format)",
    )

    # Structure generation parameters (SciLink)
    supercell_x: Optional[int] = Field(
        default=1, description="Supercell x dimension (for structure generation, default 1)"
    )
    supercell_y: Optional[int] = Field(
        default=1, description="Supercell y dimension (for structure generation, default 1)"
    )
    supercell_z: Optional[int] = Field(
        default=1, description="Supercell z dimension (for structure generation, default 1)"
    )
    structure_size: Optional[tuple[int, int, int]] = Field(
        default=None, description="Supercell dimensions (x, y, z)"
    )
    surface_miller_indices: Optional[tuple[int, int, int]] = Field(
        default=(1, 0, 0), description="Miller indices for surface (e.g., (1,1,1), default (1,0,0))"
    )
    vacuum_thickness: Optional[float] = Field(
        default=15.0, description="Vacuum thickness in Angstrom (default 15.0)"
    )
    vacuum_size: Optional[float] = Field(
        default=None,
        description="Vacuum size in Angstrom (alternative to vacuum_thickness)",
    )
    use_scilink: bool = Field(
        default=False, description="Whether to use SciLink for structure generation"
    )
    output_format: Optional[str] = Field(
        default="xyz", description="Output format (xyz, poscar, cif, pdb)"
    )
    relax: bool = Field(
        default=True, description="Whether to relax the structure after generation"
    )

    # AFM/STM-specific parameters
    tip_height: Optional[float] = Field(
        default=None,
        description="Tip height in Angstrom (AFM/STM)",
    )
    scan_size: Optional[tuple[float, float]] = Field(
        default=None,
        description="Scan size in nm (x, y) (AFM/STM)",
    )

    # AFM ppafm-specific parameters
    afm_pix_per_angstrome: Optional[int] = Field(
        default=None,
        description="AFM grid points per Ångström",
    )
    afm_scan_dim: Optional[tuple[int, int, int]] = Field(
        default=None,
        description="AFM scan dimensions (points in x, y, z)",
    )
    afm_scan_window_min: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="AFM scan window minimum coordinates [Å]",
    )
    afm_scan_window_max: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="AFM scan window maximum coordinates [Å]",
    )
    afm_i_zpp: Optional[int] = Field(
        default=None,
        description="AFM probe atomic number (8=O for CO tip)",
    )
    afm_qs: Optional[list[float]] = Field(
        default=None,
        description="AFM tip charge magnitudes [e]",
    )
    afm_qzs: Optional[list[float]] = Field(
        default=None,
        description="AFM tip charge positions [Å]",
    )
    afm_sigma: Optional[float] = Field(
        default=None,
        description="AFM Gaussian width [Å]",
    )
    afm_a_pauli: Optional[float] = Field(
        default=None,
        description="AFM Pauli repulsion prefactor",
    )
    afm_b_pauli: Optional[float] = Field(
        default=None,
        description="AFM Pauli repulsion exponent",
    )
    afm_fdbm_vdw_type: Optional[str] = Field(
        default=None,
        description="AFM van der Waals type (D3 for DFT-D3)",
    )
    afm_d3_params: Optional[str] = Field(
        default=None,
        description="AFM DFT-D3 parameters (PBE, BLYP, etc.)",
    )
    afm_lj_vdw_damp: Optional[int] = Field(
        default=None,
        description="AFM Lennard-Jones damping",
    )
    afm_df_steps: Optional[int] = Field(
        default=None,
        description="AFM oscillation amplitude steps",
    )
    afm_tip_r0: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="AFM equilibrium tip position [Å]",
    )
    afm_tip_stiffness: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="AFM spring constants [N/m]",
    )
    afm_npbc: Optional[tuple[int, int, int]] = Field(
        default=None,
        description="AFM periodic boundary conditions",
    )
    afm_f0_cantilever: Optional[float] = Field(
        default=None,
        description="AFM cantilever frequency [Hz]",
    )
    afm_k_cantilever: Optional[float] = Field(
        default=None,
        description="AFM cantilever stiffness [N/m]",
    )
    afm_colorscale: Optional[str] = Field(
        default=None,
        description="AFM output colorscale",
    )
    afm_minimize_memory: Optional[bool] = Field(
        default=None,
        description="AFM memory optimization flag",
    )

    # STM-specific parameters
    bias_voltage: Optional[float] = Field(
        default=None,
        description="Bias voltage in V (STM)",
    )
    stm_symmetries: Optional[list[int]] = Field(
        default=None,
        description="STM surface symmetries (list of 0, 1, 2)",
    )
    stm_use_density: Optional[bool] = Field(
        default=None,
        description="Use electron density instead of LDOS",
    )
    stm_repeat_x: Optional[int] = Field(
        default=None,
        description="STM x-direction scan repeat",
    )
    stm_repeat_y: Optional[int] = Field(
        default=None,
        description="STM y-direction scan repeat",
    )
    stm_z0: Optional[float] = Field(
        default=None,
        description="STM initial z position for constant current scan",
    )
    stm_sts_bias_start: Optional[float] = Field(
        default=None,
        description="STM STS bias start (eV)",
    )
    stm_sts_bias_end: Optional[float] = Field(
        default=None,
        description="STM STS bias end (eV)",
    )
    stm_sts_bias_step: Optional[float] = Field(
        default=None,
        description="STM STS bias step (eV)",
    )
    stm_sts_x: Optional[float] = Field(
        default=None,
        description="STM STS x position (Angstrom)",
    )
    stm_sts_y: Optional[float] = Field(
        default=None,
        description="STM STS y position (Angstrom)",
    )
    stm_linescan_npoints: Optional[int] = Field(
        default=None,
        description="STM line scan number of points",
    )
    stm_gpaw_mode: Optional[str] = Field(
        default=None,
        description="GPAW calculation mode (pw, lcao, fd, etc.)",
    )
    stm_gpaw_kpts: Optional[tuple[int, int, int]] = Field(
        default=None,
        description="GPAW k-point grid",
    )
    stm_gpaw_symmetry: Optional[str] = Field(
        default=None,
        description="GPAW symmetry (on or off)",
    )
    stm_gpaw_xc: Optional[str] = Field(
        default=None,
        description="GPAW exchange-correlation functional",
    )
    stm_gpaw_h: Optional[float] = Field(
        default=None,
        description="GPAW grid spacing (Angstrom)",
    )

    # IETS-specific parameters
    energy_range: Optional[tuple[float, float]] = Field(
        default=None,
        description="Energy range in meV (min, max) (IETS)",
    )
    iets_voltage: Optional[float] = Field(
        default=None,
        description="IETS voltage (eV)",
    )
    iets_work_function: Optional[float] = Field(
        default=None,
        description="IETS work function (eV)",
    )
    iets_eta: Optional[float] = Field(
        default=None,
        description="IETS energy smearing (eV)",
    )
    iets_amplitude: Optional[float] = Field(
        default=None,
        description="IETS vibration amplitude",
    )
    iets_s_orbital: Optional[float] = Field(
        default=None,
        description="IETS s-orbital coefficient",
    )
    iets_px_orbital: Optional[float] = Field(
        default=None,
        description="IETS px-orbital coefficient",
    )
    iets_py_orbital: Optional[float] = Field(
        default=None,
        description="IETS py-orbital coefficient",
    )
    iets_pz_orbital: Optional[float] = Field(
        default=None,
        description="IETS pz-orbital coefficient",
    )
    iets_dxz_orbital: Optional[float] = Field(
        default=None,
        description="IETS dxz-orbital coefficient",
    )
    iets_dyz_orbital: Optional[float] = Field(
        default=None,
        description="IETS dyz-orbital coefficient",
    )
    iets_dz2_orbital: Optional[float] = Field(
        default=None,
        description="IETS dz2-orbital coefficient",
    )
    iets_gpaw_file: Optional[str] = Field(
        default=None,
        description="IETS GPAW calculation file path",
    )
    iets_sample_orbs: Optional[str] = Field(
        default=None,
        description="IETS sample orbitals (sp or spd)",
    )
    iets_pbc: Optional[tuple[int, int]] = Field(
        default=None,
        description="IETS periodic boundary conditions",
    )
    iets_fermi: Optional[float] = Field(
        default=None,
        description="IETS Fermi level",
    )
    iets_cut_min: Optional[float] = Field(
        default=None,
        description="IETS energy cutoff minimum (eV)",
    )
    iets_cut_max: Optional[float] = Field(
        default=None,
        description="IETS energy cutoff maximum (eV)",
    )
    iets_ncpu: Optional[int] = Field(
        default=None,
        description="IETS number of CPU cores",
    )
    iets_x_range: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="IETS x-grid (xmin, xmax, dx)",
    )
    iets_y_range: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="IETS y-grid (ymin, ymax, dy)",
    )
    iets_z_range: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="IETS z-grid (zmin, zmax, dz)",
    )
    iets_charge_q: Optional[float] = Field(
        default=None,
        description="IETS tip charge",
    )
    iets_stiffness_k: Optional[float] = Field(
        default=None,
        description="IETS tip stiffness (N/m)",
    )
    iets_effective_mass: Optional[float] = Field(
        default=None,
        description="IETS effective mass (Atomic Units)",
    )

    # General metadata
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the parsing (0-1)",
        ge=0.0,
        le=1.0,
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="List of ambiguous elements in the query",
    )
    missing_parameters: list[str] = Field(
        default_factory=list, description="List of missing required parameters"
    )
