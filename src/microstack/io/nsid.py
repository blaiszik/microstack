"""NSID format export utilities for microscopy data.

NSID (N-Dimensional Spectroscopy and Imaging Data) is a modern HDF5-based format
for storing microscopy data with proper dimension labels and metadata.

This module uses:
- sidpy: Core data structures (Dataset, Dimension)
- pyNSID: I/O operations for writing sidpy.Dataset to NSID-compliant HDF5

This makes MicroStack output compatible with the pycroscopy analysis ecosystem.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Check if sidpy and pyNSID are available
try:
    import sidpy
    import pyNSID
    import h5py

    SIDPY_AVAILABLE = True
except ImportError:
    SIDPY_AVAILABLE = False
    sidpy = None
    pyNSID = None
    h5py = None


def _check_sidpy() -> None:
    """Raise error if sidpy/pyNSID is not available."""
    if not SIDPY_AVAILABLE:
        raise ImportError(
            "sidpy and pyNSID are required for NSID export. "
            "Install with: pip install sidpy pyNSID"
        )


def _create_sidpy_dataset(
    data: np.ndarray,
    name: str,
    title: str,
    units: str,
    quantity: str,
    data_type: str,
    dimensions: list,
    metadata: Optional[Dict[str, Any]] = None,
) -> "sidpy.Dataset":
    """Create a sidpy Dataset with proper dimensions and metadata.

    Args:
        data: Numpy array of data
        name: Dataset name (used for HDF5 path)
        title: Human-readable title for the dataset
        units: Physical units of the data
        quantity: Physical quantity (e.g., 'Height', 'Current')
        data_type: sidpy data type ('image', 'spectrum', 'image_stack', etc.)
        dimensions: List of dimension specs, each being:
            (values, name, units, quantity, dimension_type)
        metadata: Optional metadata dictionary

    Returns:
        Configured sidpy.Dataset
    """
    _check_sidpy()

    dataset = sidpy.Dataset.from_array(data, name=name)
    dataset.units = units
    dataset.quantity = quantity
    dataset.title = title
    
    # Use sidpy's DataType enum for proper typing
    # Map string to sidpy.DataType enum
    data_type_map = {
        'image': sidpy.DataType.IMAGE,
        'spectrum': sidpy.DataType.SPECTRUM,
        'image_stack': sidpy.DataType.IMAGE_STACK,
        'spectral_image': sidpy.DataType.SPECTRAL_IMAGE,
        'image_4d': sidpy.DataType.IMAGE_4D,
        'unknown': sidpy.DataType.UNKNOWN,
    }
    dataset.data_type = data_type_map.get(data_type.lower(), sidpy.DataType.UNKNOWN)

    # Set dimensions with proper dimension_type enum
    dim_type_map = {
        'spatial': sidpy.DimensionType.SPATIAL,
        'spectral': sidpy.DimensionType.SPECTRAL,
        'temporal': sidpy.DimensionType.TEMPORAL,
        'reciprocal': sidpy.DimensionType.RECIPROCAL,
        'unknown': sidpy.DimensionType.UNKNOWN,
    }
    
    for i, dim_spec in enumerate(dimensions):
        values, dim_name, dim_units, dim_quantity, dim_type = dim_spec
        dim = sidpy.Dimension(values, name=dim_name, units=dim_units, quantity=dim_quantity)
        dim.dimension_type = dim_type_map.get(dim_type.lower(), sidpy.DimensionType.UNKNOWN)
        dataset.set_dimension(i, dim)

    # Attach metadata
    if metadata:
        dataset.metadata = metadata

    return dataset


def save_stm_to_nsid(
    filepath: str | Path,
    constant_current_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    constant_height_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    sts_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export STM data to NSID format using pyNSID.

    Args:
        filepath: Output HDF5 file path
        constant_current_data: Tuple of (x, y, height) arrays for constant current scan
        constant_height_data: Tuple of (x, y, current) arrays for constant height scan
        sts_data: Tuple of (bias, I, dIdV) arrays for STS spectrum
        metadata: Dictionary of metadata (bias_voltage, tip_height, etc.)

    Returns:
        Path to the created file
    """
    _check_sidpy()

    filepath = Path(filepath)
    metadata = metadata or {}

    with h5py.File(filepath, "w") as h5f:
        # Add file-level attributes
        h5f.attrs["file_format"] = "NSID"
        h5f.attrs["software"] = "MicroStack"
        h5f.attrs["data_source"] = "simulation"

        # Create measurement group
        meas_group = h5f.create_group("Measurement_000")

        # Constant Current scan
        if constant_current_data is not None:
            x, y, height = constant_current_data

            # Extract 1D dimension arrays
            if x.ndim == 2:
                x_vals = x[0, :]
                y_vals = y[:, 0]
            else:
                x_vals = x
                y_vals = y

            cc_dataset = _create_sidpy_dataset(
                data=height,
                name="STM_Constant_Current",
                title="STM Constant Current Image",
                units="Å",
                quantity="Height",
                data_type="image",
                dimensions=[
                    (y_vals, "y", "Å", "Length", "spatial"),
                    (x_vals, "x", "Å", "Length", "spatial"),
                ],
                metadata=metadata,
            )

            # Use pyNSID to write the dataset
            pyNSID.io.write_nsid_dataset(cc_dataset, meas_group, main_data_name="Constant_Current")

        # Constant Height scan
        if constant_height_data is not None:
            x, y, current = constant_height_data

            if x.ndim == 2:
                x_vals = x[0, :]
                y_vals = y[:, 0]
            else:
                x_vals = x
                y_vals = y

            ch_dataset = _create_sidpy_dataset(
                data=current,
                name="STM_Constant_Height",
                title="STM Constant Height Image",
                units="nA",
                quantity="Current",
                data_type="image",
                dimensions=[
                    (y_vals, "y", "Å", "Length", "spatial"),
                    (x_vals, "x", "Å", "Length", "spatial"),
                ],
                metadata=metadata,
            )

            pyNSID.io.write_nsid_dataset(ch_dataset, meas_group, main_data_name="Constant_Height")

        # STS spectrum
        if sts_data is not None:
            bias, I, dIdV = sts_data

            # I(V) spectrum
            sts_I_dataset = _create_sidpy_dataset(
                data=I,
                name="STS_Current",
                title="STS I(V) Spectrum",
                units="nA",
                quantity="Current",
                data_type="spectrum",
                dimensions=[
                    (bias, "bias", "V", "Voltage", "spectral"),
                ],
                metadata=metadata,
            )
            pyNSID.io.write_nsid_dataset(sts_I_dataset, meas_group, main_data_name="STS_Current")

            # dI/dV spectrum
            sts_dIdV_dataset = _create_sidpy_dataset(
                data=dIdV,
                name="STS_dIdV",
                title="STS dI/dV Spectrum",
                units="nA/V",
                quantity="Conductance",
                data_type="spectrum",
                dimensions=[
                    (bias, "bias", "V", "Voltage", "spectral"),
                ],
                metadata=metadata,
            )
            pyNSID.io.write_nsid_dataset(sts_dIdV_dataset, meas_group, main_data_name="STS_dIdV")

    return filepath


def save_afm_to_nsid(
    filepath: str | Path,
    afm_image: Optional[np.ndarray] = None,
    height_map: Optional[np.ndarray] = None,
    vdw_spheres: Optional[np.ndarray] = None,
    atomic_disks: Optional[np.ndarray] = None,
    es_map: Optional[np.ndarray] = None,
    scan_window: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export AFM data to NSID format using pyNSID.

    Args:
        filepath: Output HDF5 file path
        afm_image: 2D AFM frequency shift image
        height_map: 2D height map
        vdw_spheres: 2D vdW spheres map
        atomic_disks: 2D atomic disks map
        es_map: 2D electrostatic map
        scan_window: ((x_min, y_min), (x_max, y_max)) in Angstroms
        metadata: Dictionary of metadata

    Returns:
        Path to the created file
    """
    _check_sidpy()

    filepath = Path(filepath)
    metadata = metadata or {}

    # Default scan window if not provided
    if scan_window is None:
        scan_window = ((0, 0), (10, 10))

    with h5py.File(filepath, "w") as h5f:
        h5f.attrs["file_format"] = "NSID"
        h5f.attrs["software"] = "MicroStack"
        h5f.attrs["data_source"] = "simulation"

        meas_group = h5f.create_group("Measurement_000")

        def _save_afm_channel(data: np.ndarray, name: str, title: str, units: str, quantity: str):
            """Helper to save an AFM channel."""
            ny, nx = data.shape
            x_vals = np.linspace(scan_window[0][0], scan_window[1][0], nx)
            y_vals = np.linspace(scan_window[0][1], scan_window[1][1], ny)

            dataset = _create_sidpy_dataset(
                data=data,
                name=name,
                title=title,
                units=units,
                quantity=quantity,
                data_type="image",
                dimensions=[
                    (y_vals, "y", "Å", "Length", "spatial"),
                    (x_vals, "x", "Å", "Length", "spatial"),
                ],
                metadata=metadata,
            )
            pyNSID.io.write_nsid_dataset(dataset, meas_group, main_data_name=name)

        if afm_image is not None:
            _save_afm_channel(afm_image, "Frequency_Shift", "AFM Frequency Shift", "Hz", "Frequency")

        if height_map is not None:
            _save_afm_channel(height_map, "Height_Map", "AFM Height Map", "Å", "Height")

        if vdw_spheres is not None:
            _save_afm_channel(vdw_spheres, "vdW_Spheres", "vdW Spheres Map", "a.u.", "vdW")

        if atomic_disks is not None:
            _save_afm_channel(atomic_disks, "Atomic_Disks", "Atomic Disks Map", "a.u.", "Atoms")

        if es_map is not None:
            _save_afm_channel(es_map, "Electrostatic", "Electrostatic Potential Map", "V", "Potential")

    return filepath


def save_tem_to_nsid(
    filepath: str | Path,
    image_data: np.ndarray,
    sampling: Optional[float] = None,
    energy: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export TEM data to NSID format using pyNSID.

    Args:
        filepath: Output HDF5 file path
        image_data: 2D TEM intensity image
        sampling: Pixel sampling in Angstroms (default 0.1)
        energy: Electron energy in keV (default 200)
        metadata: Dictionary of metadata

    Returns:
        Path to the created file
    """
    _check_sidpy()

    filepath = Path(filepath)
    metadata = metadata or {}
    sampling = sampling or 0.1
    energy = energy or 200.0

    # Squeeze any singleton dimensions
    image_data = np.squeeze(image_data)
    if image_data.ndim != 2:
        raise ValueError(f"TEM image must be 2D, got shape {image_data.shape}")

    ny, nx = image_data.shape
    x_vals = np.arange(nx) * sampling
    y_vals = np.arange(ny) * sampling

    # Add TEM-specific metadata
    full_metadata = {
        "energy_keV": energy,
        "sampling_angstrom": sampling,
        **metadata,
    }

    with h5py.File(filepath, "w") as h5f:
        h5f.attrs["file_format"] = "NSID"
        h5f.attrs["software"] = "MicroStack"
        h5f.attrs["data_source"] = "simulation"

        meas_group = h5f.create_group("Measurement_000")

        dataset = _create_sidpy_dataset(
            data=image_data,
            name="TEM_Intensity",
            title=f"TEM Image ({energy:.0f} keV)",
            units="a.u.",
            quantity="Intensity",
            data_type="image",
            dimensions=[
                (y_vals, "y", "Å", "Length", "spatial"),
                (x_vals, "x", "Å", "Length", "spatial"),
            ],
            metadata=full_metadata,
        )
        pyNSID.io.write_nsid_dataset(dataset, meas_group, main_data_name="TEM_Intensity")

    return filepath


def save_iets_to_nsid(
    filepath: str | Path,
    iets_data: np.ndarray,
    x_range: Tuple[float, float, float],
    y_range: Tuple[float, float, float],
    z_range: Tuple[float, float, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export IETS data to NSID format using pyNSID.

    Args:
        filepath: Output HDF5 file path
        iets_data: 3D or 4D IETS data array
        x_range: (x_min, x_max, x_step) in Angstroms
        y_range: (y_min, y_max, y_step) in Angstroms
        z_range: (z_min, z_max, z_step) in Angstroms
        metadata: Dictionary of metadata (voltage, eta, orbital coefficients, etc.)

    Returns:
        Path to the created file
    """
    _check_sidpy()

    filepath = Path(filepath)
    metadata = metadata or {}

    # Create dimension arrays
    x_vals = np.arange(x_range[0], x_range[1], x_range[2])
    y_vals = np.arange(y_range[0], y_range[1], y_range[2])
    z_vals = np.arange(z_range[0], z_range[1], z_range[2])

    with h5py.File(filepath, "w") as h5f:
        h5f.attrs["file_format"] = "NSID"
        h5f.attrs["software"] = "MicroStack"
        h5f.attrs["data_source"] = "simulation"

        meas_group = h5f.create_group("Measurement_000")

        if iets_data.ndim == 3:
            # 3D spatial grid
            dataset = _create_sidpy_dataset(
                data=iets_data,
                name="IETS_Signal",
                title="IETS Tunneling Signal (3D)",
                units="a.u.",
                quantity="IETS Signal",
                data_type="image_stack",
                dimensions=[
                    (x_vals[: iets_data.shape[0]], "x", "Å", "Length", "spatial"),
                    (y_vals[: iets_data.shape[1]], "y", "Å", "Length", "spatial"),
                    (z_vals[: iets_data.shape[2]], "z", "Å", "Length", "spatial"),
                ],
                metadata=metadata,
            )
        elif iets_data.ndim == 4:
            # 4D data
            dataset = _create_sidpy_dataset(
                data=iets_data,
                name="IETS_Signal",
                title="IETS Tunneling Signal (4D)",
                units="a.u.",
                quantity="IETS Signal",
                data_type="image_4d",
                dimensions=[
                    (x_vals[: iets_data.shape[0]], "x", "Å", "Length", "spatial"),
                    (y_vals[: iets_data.shape[1]], "y", "Å", "Length", "spatial"),
                    (z_vals[: iets_data.shape[2]], "z", "Å", "Length", "spatial"),
                    (np.arange(iets_data.shape[3]), "index", "", "Index", "spectral"),
                ],
                metadata=metadata,
            )
        else:
            # Other dimensionalities
            dims = []
            for i in range(iets_data.ndim):
                dims.append((np.arange(iets_data.shape[i]), f"dim_{i}", "", "Index", "unknown"))
            dataset = _create_sidpy_dataset(
                data=iets_data,
                name="IETS_Signal",
                title="IETS Tunneling Signal",
                units="a.u.",
                quantity="IETS Signal",
                data_type="unknown",
                dimensions=dims,
                metadata=metadata,
            )

        pyNSID.io.write_nsid_dataset(dataset, meas_group, main_data_name="IETS_Signal")

    return filepath


def validate_nsid_file(filepath: str | Path, verbose: bool = True) -> Dict[str, Any]:
    """Validate an NSID file using pyNSID and return diagnostics.

    Args:
        filepath: Path to HDF5 file to validate
        verbose: Whether to print diagnostics

    Returns:
        Dictionary with validation results
    """
    _check_sidpy()

    filepath = Path(filepath)
    result = {
        "valid": True,
        "filepath": str(filepath),
        "datasets": [],
        "errors": [],
        "warnings": [],
    }

    if not filepath.exists():
        result["valid"] = False
        result["errors"].append(f"File not found: {filepath}")
        return result

    try:
        with h5py.File(filepath, "r") as h5f:
            # Check file-level attributes
            if "file_format" not in h5f.attrs:
                result["warnings"].append("Missing 'file_format' attribute")

            # Use pyNSID to read datasets
            def _find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Check if it's a main dataset (not a dimension scale)
                    if "quantity" in obj.attrs or "units" in obj.attrs:
                        ds_info = {
                            "name": name,
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                        }
                        for attr in ["title", "units", "quantity", "data_type"]:
                            if attr in obj.attrs:
                                val = obj.attrs[attr]
                                ds_info[attr] = val.decode() if isinstance(val, bytes) else val

                        # Check dimensions
                        ds_info["dimensions"] = []
                        if obj.dims:
                            for i, dim in enumerate(obj.dims):
                                if dim.label:
                                    ds_info["dimensions"].append(dim.label)

                        result["datasets"].append(ds_info)

            h5f.visititems(_find_datasets)

            if not result["datasets"]:
                result["warnings"].append("No NSID datasets found")

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to read file: {e}")

    result["valid"] = len(result["errors"]) == 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"NSID Validation: {filepath.name}")
        print("=" * 60)

        if result["valid"]:
            print("✅ File is valid NSID format")
        else:
            print("❌ File has errors")

        print(f"\nDatasets found: {len(result['datasets'])}")
        for ds in result["datasets"]:
            dims_str = ", ".join(ds.get("dimensions", [])) or "none"
            print(f"  📊 {ds['name']}")
            print(f"     Shape: {ds['shape']}, Type: {ds.get('data_type', 'unknown')}")
            print(f"     Units: {ds.get('units', 'unknown')}, Dims: [{dims_str}]")

        if result["warnings"]:
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for w in result["warnings"]:
                print(f"   - {w}")

        if result["errors"]:
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for e in result["errors"]:
                print(f"   - {e}")

    return result
