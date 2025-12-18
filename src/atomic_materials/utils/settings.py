"""Configuration settings for ATOMIC using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration for ATOMIC."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    google_api_key: str = Field(
        ...,
        description="Google API key for Gemini models used by SciLink (required)",
    )
    scilink_generator_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use for SciLink's StructureGenerator",
    )

    # ===== LLM Configuration =====
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model to use",
    )
    llm_agent: Literal["anthropic", "deepseek"] = Field(
        default="anthropic",
        description="Which LLM agent to use (anthropic or deepseek)",
    )
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API key for natural language query parsing",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model to use (deepseek-chat or deepseek-reasoner)",
    )

    # ===== Structure Database APIs =====
    mp_api_key: Optional[str] = Field(
        default=None,
        description="Materials Project API key (optional, required for MP access)",
    )
    oqmd_base_url: str = Field(
        default="https://oqmd.org/oqmdapi",
        description="OQMD API base URL",
    )

    # ===== GPU Configuration =====
    gpu_backend: Optional[Literal["cuda", "cpu"]] = Field(
        default=None,
        description="GPU backend to use (auto-detect if None)",
    )
    cuda_device_id: int = Field(
        default=0,
        description="CUDA device ID for NVIDIA GPUs",
    )

    # ===== Structure Generation Defaults =====
    default_supercell_x: int = Field(
        default=1,
        description="Default supercell x dimension if not specified by user",
    )
    default_supercell_y: int = Field(
        default=1,
        description="Default supercell y dimension if not specified by user",
    )
    default_supercell_z: int = Field(
        default=1,
        description="Default supercell z dimension if not specified by user",
    )
    default_surface_miller_indices: tuple[int, int, int] = Field(
        default=(1, 0, 0),
        description="Default Miller indices (e.g., (1,0,0) for 100 surface) if not specified",
    )
    default_vacuum_thickness: float = Field(
        default=15.0,
        description="Default vacuum thickness in Angstrom if not specified",
    )

    # ===== Output Configuration =====
    output_dir: Path = Field(
        default=Path("./atomic_output"),
        description="Directory for saving simulation outputs",
    )
    image_dpi: int = Field(
        default=300,
        description="DPI for PNG image exports",
    )

    # ===== AFM Default Parameters (ppafm AFMulator) =====
    afm_pix_per_angstrome: int = Field(
        default=10,
        description="AFM grid points per Ångström",
    )
    afm_scan_dim: tuple[int, int, int] = Field(
        default=(128, 128, 30),
        description="AFM scan dimensions (points in x, y, z)",
    )
    afm_scan_window_min: tuple[float, float, float] = Field(
        default=(2.0, 2.0, 7.0),
        description="AFM scan window minimum coordinates [Å]",
    )
    afm_scan_window_max: tuple[float, float, float] = Field(
        default=(18.0, 18.0, 10.0),
        description="AFM scan window maximum coordinates [Å]",
    )
    afm_i_zpp: int = Field(
        default=8,
        description="AFM probe atomic number (8=O for CO tip)",
    )
    afm_qs: list[float] = Field(
        default=[-10, 20, -10, 0],
        description="AFM tip charge magnitudes [e]",
    )
    afm_qzs: list[float] = Field(
        default=[0.1, 0, -0.1, 0],
        description="AFM tip charge positions [Å]",
    )
    afm_sigma: float = Field(
        default=0.71,
        description="AFM Gaussian width [Å]",
    )
    afm_a_pauli: float = Field(
        default=18.0,
        description="AFM Pauli repulsion prefactor",
    )
    afm_b_pauli: float = Field(
        default=1.0,
        description="AFM Pauli repulsion exponent",
    )
    afm_fdbm_vdw_type: str = Field(
        default="D3",
        description="AFM van der Waals type (D3 for DFT-D3)",
    )
    afm_d3_params: str = Field(
        default="PBE",
        description="AFM DFT-D3 parameters (PBE, BLYP, etc.)",
    )
    afm_lj_vdw_damp: int = Field(
        default=2,
        description="AFM Lennard-Jones damping",
    )
    afm_df_steps: int = Field(
        default=10,
        description="AFM oscillation amplitude steps",
    )
    afm_tip_r0: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 3.0),
        description="AFM equilibrium tip position [Å]",
    )
    afm_tip_stiffness: tuple[float, float, float, float] = Field(
        default=(0.25, 0.25, 0.0, 30.0),
        description="AFM spring constants [N/m]",
    )
    afm_npbc: tuple[int, int, int] = Field(
        default=(1, 1, 0),
        description="AFM periodic boundary conditions",
    )
    afm_f0_cantilever: float = Field(
        default=30300.0,
        description="AFM cantilever frequency [Hz]",
    )
    afm_k_cantilever: float = Field(
        default=1800.0,
        description="AFM cantilever stiffness [N/m]",
    )
    afm_colorscale: str = Field(
        default="gray",
        description="AFM output colorscale",
    )
    afm_minimize_memory: bool = Field(
        default=False,
        description="AFM memory optimization flag",
    )

    # ===== STM Default Parameters (ASE/GPAW-based) =====
    stm_bias_voltage: float = Field(
        default=1.0,
        description="STM bias voltage in V",
    )
    stm_tip_height: float = Field(
        default=8.0,
        description="STM tip height in Angstrom",
    )
    stm_scan_size: tuple[float, float] = Field(
        default=(10.0, 10.0),
        description="STM scan size in nm (x, y)",
    )
    stm_symmetries: list[int] = Field(
        default=[0, 1, 2],
        description="STM surface symmetries (list of 0, 1, 2)",
    )
    stm_use_density: bool = Field(
        default=False,
        description="Use electron density instead of LDOS",
    )
    stm_repeat_x: int = Field(
        default=3,
        description="STM x-direction scan repeat",
    )
    stm_repeat_y: int = Field(
        default=5,
        description="STM y-direction scan repeat",
    )
    stm_z0: Optional[float] = Field(
        default=None,
        description="STM initial z position for constant current scan",
    )
    stm_sts_bias_start: float = Field(
        default=-2.0,
        description="STM STS bias start (eV)",
    )
    stm_sts_bias_end: float = Field(
        default=2.0,
        description="STM STS bias end (eV)",
    )
    stm_sts_bias_step: float = Field(
        default=0.05,
        description="STM STS bias step (eV)",
    )
    stm_sts_x: float = Field(
        default=0.0,
        description="STM STS x position (Angstrom)",
    )
    stm_sts_y: float = Field(
        default=0.0,
        description="STM STS y position (Angstrom)",
    )
    stm_linescan_npoints: int = Field(
        default=50,
        description="STM line scan number of points",
    )
    stm_gpaw_mode: str = Field(
        default="pw",
        description="GPAW calculation mode (pw, lcao, fd, etc.)",
    )
    stm_gpaw_kpts: tuple[int, int, int] = Field(
        default=(4, 4, 1),
        description="GPAW k-point grid",
    )
    stm_gpaw_symmetry: str = Field(
        default="off",
        description="GPAW symmetry (on or off)",
    )
    stm_gpaw_xc: str = Field(
        default="LDA",
        description="GPAW exchange-correlation functional",
    )
    stm_gpaw_h: float = Field(
        default=0.2,
        description="GPAW grid spacing (Angstrom)",
    )

    # ===== IETS Default Parameters (GPAW-based) =====
    iets_voltage: float = Field(
        default=0.0,
        description="IETS voltage (energy vs. Fermi Level in eV)",
    )
    iets_work_function: float = Field(
        default=5.0,
        description="IETS work function (eV)",
    )
    iets_eta: float = Field(
        default=0.1,
        description="IETS energy smearing (eV)",
    )
    iets_amplitude: float = Field(
        default=0.05,
        description="IETS vibration amplitude",
    )
    iets_s_orbital: float = Field(
        default=1.0,
        description="IETS s-orbital coefficient",
    )
    iets_px_orbital: float = Field(
        default=0.0,
        description="IETS px-orbital coefficient",
    )
    iets_py_orbital: float = Field(
        default=0.0,
        description="IETS py-orbital coefficient",
    )
    iets_pz_orbital: float = Field(
        default=0.0,
        description="IETS pz-orbital coefficient",
    )
    iets_dxz_orbital: float = Field(
        default=0.0,
        description="IETS dxz-orbital coefficient",
    )
    iets_dyz_orbital: float = Field(
        default=0.0,
        description="IETS dyz-orbital coefficient",
    )
    iets_dz2_orbital: float = Field(
        default=0.0,
        description="IETS dz2-orbital coefficient",
    )
    iets_dft_code: str = Field(
        default="gpaw",
        description="IETS DFT code (gpaw)",
    )
    iets_gpaw_file: Optional[str] = Field(
        default=None,
        description="IETS GPAW calculation file path",
    )
    iets_sample_orbs: str = Field(
        default="sp",
        description="IETS sample orbitals (sp or spd)",
    )
    iets_pbc: tuple[int, int] = Field(
        default=(0, 0),
        description="IETS periodic boundary conditions",
    )
    iets_fermi: Optional[float] = Field(
        default=None,
        description="IETS Fermi level (None = use from file)",
    )
    iets_cut_min: float = Field(
        default=-2.5,
        description="IETS energy cutoff minimum (eV)",
    )
    iets_cut_max: float = Field(
        default=2.5,
        description="IETS energy cutoff maximum (eV)",
    )
    iets_cut_atoms: Optional[int] = Field(
        default=None,
        description="IETS number of atoms contributing to tunneling",
    )
    iets_ncpu: int = Field(
        default=4,
        description="IETS number of CPU cores for OpenMP",
    )
    iets_x_range: tuple[float, float, float] = Field(
        default=(0.0, 20.0, 0.25),
        description="IETS x-grid (xmin, xmax, dx)",
    )
    iets_y_range: tuple[float, float, float] = Field(
        default=(0.0, 15.0, 0.25),
        description="IETS y-grid (ymin, ymax, dy)",
    )
    iets_z_range: tuple[float, float, float] = Field(
        default=(10.0, 12.0, 0.1),
        description="IETS z-grid (zmin, zmax, dz)",
    )
    iets_charge_q: float = Field(
        default=0.0,
        description="IETS tip charge (PP-AFM)",
    )
    iets_stiffness_k: float = Field(
        default=0.5,
        description="IETS tip stiffness (N/m)",
    )
    iets_effective_mass: float = Field(
        default=16.0,
        description="IETS effective mass of vibrating molecule (Atomic Units)",
    )
    iets_data_format: str = Field(
        default="npy",
        description="IETS data format (npy or xsf)",
    )
    iets_plot_results: bool = Field(
        default=True,
        description="IETS plot results flag",
    )

    # ===== Logging Configuration =====
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Optional[Path] = Field(
        default=Path("./atomic.log"),
        description="Log file path (None to disable file logging)",
    )
    log_to_console: bool = Field(
        default=True,
        description="Enable console logging",
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging",
    )

    # ===== Performance Configuration =====
    num_workers: int = Field(
        default=4,
        description="Number of parallel workers for simulations",
        ge=1,
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bars during simulations",
    )
    cache_dir: Path = Field(
        default=Path("./.atomic_cache"),
        description="Cache directory for downloaded structures",
    )

    # ===== Advanced Configuration =====
    api_timeout: int = Field(
        default=30,
        description="API request timeout in seconds",
        ge=1,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for API calls",
        ge=0,
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, stack traces)",
    )

    def model_post_init(self, __context: any) -> None:
        """Create necessary directories after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file and self.log_to_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
