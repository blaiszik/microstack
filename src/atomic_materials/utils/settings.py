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

    # ===== LLM Configuration =====
    deepseek_api_key: str = Field(
        ...,
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

    # ===== Output Configuration =====
    output_dir: Path = Field(
        default=Path("./atomic_output"),
        description="Directory for saving simulation outputs",
    )
    image_dpi: int = Field(
        default=300,
        description="DPI for PNG image exports",
    )

    # ===== TEM Default Parameters =====
    tem_acceleration_voltage: float = Field(
        default=300.0,
        description="Default TEM acceleration voltage in kV",
    )
    tem_defocus: float = Field(
        default=0.0,
        description="Default TEM defocus in nm",
    )
    tem_thickness: float = Field(
        default=10.0,
        description="Default sample thickness in nm",
    )
    tem_detector_angles: tuple[float, float] = Field(
        default=(0.0, 50.0),
        description="Default TEM detector angles in mrad (min, max)",
    )
    tem_resolution: float = Field(
        default=0.05,
        description="Default TEM resolution in Angstrom",
    )

    # ===== AFM Default Parameters =====
    afm_tip_radius: float = Field(
        default=2.0,
        description="Default AFM tip radius in Angstrom",
    )
    afm_tip_stiffness: tuple[float, float, float] = Field(
        default=(0.5, 0.5, 20.0),
        description="Default AFM tip stiffness in N/m (x, y, z)",
    )
    afm_scan_height: float = Field(
        default=5.0,
        description="Default AFM scan height in Angstrom",
    )
    afm_scan_size: tuple[float, float] = Field(
        default=(10.0, 10.0),
        description="Default AFM scan size in nm (x, y)",
    )

    # ===== STM Default Parameters =====
    stm_bias_voltage: float = Field(
        default=0.1,
        description="Default STM bias voltage in V",
    )
    stm_tip_height: float = Field(
        default=5.0,
        description="Default STM tip height in Angstrom",
    )
    stm_scan_size: tuple[float, float] = Field(
        default=(10.0, 10.0),
        description="Default STM scan size in nm (x, y)",
    )

    # ===== IETS Default Parameters =====
    iets_energy_min: float = Field(
        default=0.0,
        description="Default IETS minimum energy in meV",
    )
    iets_energy_max: float = Field(
        default=500.0,
        description="Default IETS maximum energy in meV",
    )
    iets_energy_step: float = Field(
        default=1.0,
        description="Default IETS energy step in meV",
    )

    # ===== TERS Default Parameters =====
    ters_laser_wavelength: float = Field(
        default=532.0,
        description="Default TERS laser wavelength in nm",
    )
    ters_laser_power: float = Field(
        default=1.0,
        description="Default TERS laser power in mW",
    )
    ters_scan_size: tuple[float, float] = Field(
        default=(10.0, 10.0),
        description="Default TERS scan size in nm (x, y)",
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
