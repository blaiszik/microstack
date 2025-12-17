"""Report generator with AI-powered scientific discussion.

Generates markdown reports for surface relaxation analysis,
including Claude-generated scientific interpretation.
"""

import json
from datetime import datetime
from typing import Optional

from atomic_materials.utils import config


def generate_discussion(
    element: str,
    face: str,
    analysis: dict,
) -> str:
    """
    Generate scientific discussion using Claude API.

    Args:
        element: Chemical symbol
        face: Surface face
        analysis: Full analysis dictionary from comparison.full_analysis()

    Returns:
        Markdown-formatted discussion text
    """
    client = config.get_anthropic_client()

    if client is None:
        return _generate_fallback_discussion(element, face, analysis)

    # Prepare structured data for Claude
    relaxation = analysis.get("relaxation", {})
    comparison = analysis.get("comparison", {})
    reference = analysis.get("reference", {})

    prompt = f"""Write a brief scientific discussion (2 short paragraphs, no headers) for a {element}({face}) surface relaxation study.

Data:
- Energy change: {analysis.get('energy_change_eV', 0):.4f} eV
- Max displacement: {relaxation.get('max_displacement', 0):.3f} Å
- Layer changes: {json.dumps(relaxation.get('layer_changes_percent', {}), indent=2)}

Paragraph 1: Explain the physics of why this surface relaxes this way (Smoluchowski smoothing, coordination effects, etc.).
Paragraph 2: Brief implications for catalysis or microscopy.

Be specific with numbers. No headers or bullet points. Professional tone."""

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"Warning: Claude API call failed: {e}")
        return _generate_fallback_discussion(element, face, analysis)


def _generate_fallback_discussion(element: str, face: str, analysis: dict) -> str:
    """Generate a basic discussion when Claude API is unavailable."""
    relaxation = analysis.get("relaxation", {})
    comparison = analysis.get("comparison", {})
    layer_changes = relaxation.get("layer_changes_percent", {})

    d12 = layer_changes.get("d12_change", 0)

    # Determine relaxation type
    if d12 < -1:
        relax_type = "inward contraction"
        physics = (
            "reduced coordination of surface atoms leading to Smoluchowski smoothing"
        )
    elif d12 > 1:
        relax_type = "outward expansion"
        physics = "charge redistribution effects at the surface"
    else:
        relax_type = "minimal relaxation"
        physics = "near-bulk termination"

    text = f"""The {element}({face}) surface exhibits {relax_type} of the topmost layer,
consistent with {physics}. The MACE-MP potential predicts a d₁₂ spacing change of
{d12:+.1f}%, which """

    if comparison.get("has_reference"):
        agreement = comparison.get("overall_agreement", "unknown")
        ref_source = comparison.get("reference_source", "literature")
        text += f"shows {agreement} agreement with {ref_source} reference data. "
    else:
        text += "could not be compared due to lack of reference data for this surface. "

    text += f"""

The energy lowering of {analysis.get('energy_change_eV', 0):.3f} eV upon relaxation
indicates the system reaches a more stable configuration. The maximum atomic displacement
of {relaxation.get('max_displacement', 0):.3f} Å is within the typical range for
metal surface relaxations.

These surface properties are relevant for understanding catalytic activity,
epitaxial growth, and surface microscopy of {element}-based materials."""

    return text


def generate_full_report(
    element: str, face: str, analysis: dict, figure_paths: Optional[list[str]] = None
) -> str:
    """
    Generate complete markdown report.

    Args:
        element: Chemical symbol
        face: Surface face
        analysis: Full analysis dictionary
        figure_paths: List of paths to figures to include

    Returns:
        Complete markdown report as string
    """
    relaxation = analysis.get("relaxation", {})
    comparison = analysis.get("comparison", {})
    reference = analysis.get("reference", {})
    bulk = reference.get("bulk", {})
    surface_ref = reference.get("surface", {})

    # Generate discussion
    discussion = generate_discussion(element, face, analysis)

    # Build report
    report = []

    # Header
    report.append(f"# Surface Relaxation Analysis: {element}({face})")
    report.append("")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    report.append("")

    # Summary box
    agreement = comparison.get("overall_agreement", "N/A")
    agreement_emoji = {
        "excellent": "✅",
        "good": "✓",
        "moderate": "⚠️",
        "poor": "❌",
    }.get(agreement, "•")

    report.append("## Summary")
    report.append("")
    report.append(
        f"MACE-MP machine learning potential was used to predict the atomic relaxation of {element}({face}). "
    )

    if comparison.get("has_reference"):
        report.append(
            f"Results show **{agreement}** agreement with experimental/DFT reference data "
        )
        report.append(f"(mean deviation: {comparison.get('mean_deviation', 'N/A')}%).")
    else:
        report.append("No reference data available for quantitative comparison.")
    report.append("")

    # Methodology
    report.append("## Methodology")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| ML Potential | MACE-MP-0 medium |")
    report.append("| Training Data | Materials Project DFT |")
    report.append(
        f"| Surface Model | {relaxation.get('n_atoms', 'N/A')} atoms, {relaxation.get('n_layers', 'N/A')} layers |"
    )
    report.append("| Vacuum | 10 Å |")
    report.append("| Optimizer | FIRE |")
    report.append("")

    # Bulk Properties
    if bulk:
        report.append("## Bulk Properties")
        report.append("")
        report.append(f"*Source: {bulk.get('source', 'N/A')}*")
        report.append("")
        report.append("| Property | Value |")
        report.append("|----------|-------|")
        if bulk.get("lattice_constant"):
            report.append(f"| Lattice constant | {bulk['lattice_constant']:.3f} Å |")
        if bulk.get("crystal_system"):
            report.append(f"| Crystal system | {bulk['crystal_system']} |")
        if bulk.get("space_group"):
            report.append(f"| Space group | {bulk['space_group']} |")
        if bulk.get("mp_id"):
            report.append(f"| Materials Project ID | {bulk['mp_id']} |")
        report.append("")

    # Results
    report.append("## Results")
    report.append("")

    # Energy
    report.append("### Energy")
    report.append("")
    report.append("| State | Energy (eV) |")
    report.append("|-------|-------------|")
    report.append(
        f"| Initial (unrelaxed) | {analysis.get('initial_energy_eV', 0):.4f} |"
    )
    report.append(f"| Final (relaxed) | {analysis.get('final_energy_eV', 0):.4f} |")
    report.append(f"| **Change** | **{analysis.get('energy_change_eV', 0):.4f}** |")
    report.append("")

    # Surface Relaxation
    report.append("### Surface Relaxation")
    report.append("")

    layer_changes = relaxation.get("layer_changes_percent", {})

    if comparison.get("has_reference") and comparison.get("layer_comparisons"):
        report.append("| Layer Spacing | MACE Prediction | Reference | Deviation |")
        report.append("|---------------|-----------------|-----------|-----------|")

        for lc in comparison["layer_comparisons"]:
            layer = lc["layer"].replace(
                "d", "d₁₂" if "12" in lc["layer"] else lc["layer"]
            )
            layer = layer.replace("12", "₁₂").replace("23", "₂₃").replace("34", "₃₄")
            report.append(
                f"| {layer} | {lc['ml_prediction']:+.1f}% | {lc['reference']:+.1f}% | {lc['deviation']:.1f}% |"
            )

        report.append("")
        report.append(f"**Overall Agreement**: {agreement_emoji} {agreement.upper()}")
        report.append(f"  \n{comparison.get('agreement_description', '')}")
    else:
        report.append("| Layer Spacing | Change (%) |")
        report.append("|---------------|------------|")
        for key, value in layer_changes.items():
            layer = key.replace("_change", "").replace(
                "d", "d₁₂" if "12" in key else key
            )
            layer = layer.replace("12", "₁₂").replace("23", "₂₃").replace("34", "₃₄")
            report.append(f"| {layer} | {value:+.2f} |")

    report.append("")

    # Atomic Displacements
    report.append("### Atomic Displacements")
    report.append("")
    report.append(
        f"- Maximum displacement: **{relaxation.get('max_displacement', 0):.3f} Å**"
    )
    report.append(
        f"- Mean displacement: {relaxation.get('mean_displacement', 0):.3f} Å"
    )
    report.append(
        f"- Maximum z-displacement: {relaxation.get('max_z_displacement', 0):.3f} Å"
    )
    report.append("")

    # Figures
    if figure_paths:
        report.append("## Visualization")
        report.append("")
        for fig_path in figure_paths:
            report.append(f"![Surface Relaxation]({fig_path})")
            report.append("")

    # Discussion
    report.append("## Discussion")
    report.append("")
    report.append(discussion)
    report.append("")

    # References
    report.append("## References")
    report.append("")

    ref_num = 1
    if bulk.get("mp_id"):
        report.append(
            f"{ref_num}. Materials Project: [{bulk['mp_id']}](https://materialsproject.org/materials/{bulk['mp_id']})"
        )
        ref_num += 1

    if surface_ref and surface_ref.get("source"):
        report.append(f"{ref_num}. {surface_ref['source']}")
        ref_num += 1

    report.append(
        f'{ref_num}. Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks" (2022)'
    )

    report.append("")
    report.append("---")
    report.append("*Generated by ATOMIC AI Materials Scientist*")

    return "\n".join(report)


if __name__ == "__main__":
    print("Report generator module loaded successfully")
    print("\nAvailable functions:")
    print("  - generate_discussion(element, face, analysis)")
    print("  - generate_full_report(element, face, analysis, figure_paths)")
