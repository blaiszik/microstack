"""Custom exceptions for Microscopy CLI."""


class MicroscopyCliError(Exception):
    """Base exception for all Microscopy CLI errors."""

    pass


# ===== Configuration Errors =====


class ConfigurationError(MicroscopyCliError):
    """Raised when there's a configuration error."""

    pass


class APIKeyMissingError(ConfigurationError):
    """Raised when a required API key is missing."""

    def __init__(self, api_name: str):
        super().__init__(
            f"API key for {api_name} is missing. "
            f"Please set it in your .env file or environment variables."
        )
        self.api_name = api_name


# ===== Structure Retrieval Errors =====


class StructureError(MicroscopyCliError):
    """Base exception for structure-related errors."""

    pass


class StructureNotFoundError(StructureError):
    """Raised when a structure cannot be found in the database."""

    def __init__(self, identifier: str, source: str):
        super().__init__(
            f"Structure '{identifier}' not found in {source}. "
            f"Please check the identifier and try again."
        )
        self.identifier = identifier
        self.source = source


class StructureLoadError(StructureError):
    """Raised when a structure file cannot be loaded."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Failed to load structure from '{file_path}': {reason}"
        )
        self.file_path = file_path
        self.reason = reason


class StructureConversionError(StructureError):
    """Raised when structure conversion fails."""

    def __init__(self, from_format: str, to_format: str, reason: str):
        super().__init__(
            f"Failed to convert structure from {from_format} to {to_format}: {reason}"
        )
        self.from_format = from_format
        self.to_format = to_format
        self.reason = reason


# ===== Simulation Errors =====


class SimulationError(MicroscopyCliError):
    """Base exception for simulation-related errors."""

    pass


class SimulationSetupError(SimulationError):
    """Raised when simulation setup fails."""

    def __init__(self, microscopy_type: str, reason: str):
        super().__init__(
            f"Failed to set up {microscopy_type} simulation: {reason}"
        )
        self.microscopy_type = microscopy_type
        self.reason = reason


class SimulationExecutionError(SimulationError):
    """Raised when simulation execution fails."""

    def __init__(self, microscopy_type: str, reason: str):
        super().__init__(
            f"{microscopy_type} simulation failed: {reason}"
        )
        self.microscopy_type = microscopy_type
        self.reason = reason


# ===== GPU Errors =====


class GPUError(MicroscopyCliError):
    """Base exception for GPU-related errors."""

    pass


class GPUNotAvailableError(GPUError):
    """Raised when GPU is requested but not available."""

    def __init__(self, backend: str):
        super().__init__(
            f"GPU backend '{backend}' is not available. "
            f"Falling back to CPU mode."
        )
        self.backend = backend


class GPUInitializationError(GPUError):
    """Raised when GPU initialization fails."""

    def __init__(self, backend: str, reason: str):
        super().__init__(
            f"Failed to initialize GPU backend '{backend}': {reason}"
        )
        self.backend = backend
        self.reason = reason


# ===== LLM Errors =====


class LLMError(MicroscopyCliError):
    """Base exception for LLM-related errors."""

    pass


class QueryParsingError(LLMError):
    """Raised when LLM fails to parse a query."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Failed to parse query '{query}': {reason}"
        )
        self.query = query
        self.reason = reason


class LLMConnectionError(LLMError):
    """Raised when connection to LLM API fails."""

    def __init__(self, provider: str, reason: str):
        super().__init__(
            f"Failed to connect to {provider} API: {reason}"
        )
        self.provider = provider
        self.reason = reason


# ===== Validation Errors =====


class ValidationError(MicroscopyCliError):
    """Base exception for validation errors."""

    pass


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""

    def __init__(self, parameter: str, value: any, reason: str):
        super().__init__(
            f"Invalid value for parameter '{parameter}': {value}. {reason}"
        )
        self.parameter = parameter
        self.value = value
        self.reason = reason


class MicroscopyTypeError(ValidationError):
    """Raised when an unsupported microscopy type is requested."""

    def __init__(self, microscopy_type: str, supported_types: list[str]):
        super().__init__(
            f"Unsupported microscopy type: {microscopy_type}. "
            f"Supported types: {', '.join(supported_types)}"
        )
        self.microscopy_type = microscopy_type
        self.supported_types = supported_types


# ===== Output Errors =====


class OutputError(MicroscopyCliError):
    """Base exception for output-related errors."""

    pass


class OutputSaveError(OutputError):
    """Raised when output cannot be saved."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Failed to save output to '{file_path}': {reason}"
        )
        self.file_path = file_path
        self.reason = reason


class OutputFormatError(OutputError):
    """Raised when an unsupported output format is requested."""

    def __init__(self, format: str, supported_formats: list[str]):
        super().__init__(
            f"Unsupported output format: {format}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
        self.format = format
        self.supported_formats = supported_formats
