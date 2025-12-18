"""Report generator with AI-powered scientific discussion.

Generates markdown reports for workflow execution,
including AI agent detection and task-specific summaries.
Supports structure generation, relaxation, and microscopy simulations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from microstack.utils.logging import get_logger

logger = get_logger("report_generator")


def detect_ai_agent(parsed_params: Optional[Any]) -> str:
    """
    Detect which AI agent/LLM is being used.

    Args:
        parsed_params: Parsed query parameters (not used, for backward compatibility)

    Returns:
        Name of the AI agent being used
    """
    # Check the active LLM agent from settings
    try:
        from microstack.utils.settings import settings

        llm_agent = getattr(settings, "llm_agent", "gemini").lower()

        if "anthropic" in llm_agent or "claude" in llm_agent:
            return "Claude (Anthropic)"
        elif "gemini" in llm_agent or "google" in llm_agent:
            return "Gemini (Google)"
        elif "deepseek" in llm_agent:
            return "DeepSeek"
        elif "openai" in llm_agent or "gpt" in llm_agent:
            return "GPT (OpenAI)"
    except Exception as e:
        logger.debug(f"Could not detect LLM agent from settings: {e}")

    return "Unknown AI Agent"


def generate_task_summary(state: "WorkflowState") -> str:  # noqa: F821
    """
    Generate a concise summary of completed tasks.

    Args:
        state: WorkflowState object

    Returns:
        Markdown-formatted summary
    """
    summary_lines = []

    # Completed Tasks
    if state.structure_info:
        element = state.structure_info.get("element", "Unknown")
        face = state.structure_info.get("face", "Unknown")
        formula = state.structure_info.get("formula", "Unknown")
        num_atoms = state.structure_info.get("num_atoms", 0)
        summary_lines.append(f"- **Structure Generation**: ✓ {element}({face}) - {formula} ({num_atoms} atoms)")

    if state.relaxation_results:
        energy_change = state.relaxation_results.get("energy_change", 0)
        summary_lines.append(f"- **Structure Relaxation**: ✓ ΔE = {energy_change:.4f} eV")

    if state.microscopy_results:
        for micro_type, results in state.microscopy_results.items():
            if isinstance(results, dict) and results:
                summary_lines.append(f"- **{micro_type.upper()} Simulation**: ✓ Complete")

    # Errors and Warnings
    if state.errors:
        summary_lines.append(f"- ⚠️ **Errors ({len(state.errors)})**")

    if state.warnings:
        summary_lines.append(f"- ⚠️ **Warnings ({len(state.warnings)})**")

    return "\n".join(summary_lines)


def generate_structure_section(state: "WorkflowState") -> Optional[str]:  # noqa: F821
    """Generate detailed structure generation section."""
    if not state.structure_info:
        return None

    lines = []
    lines.append("## Structure Generation")
    lines.append("")

    struct_info = state.structure_info
    lines.append(f"**Element**: {struct_info.get('element', 'N/A')}")
    lines.append(f"**Surface Face**: {struct_info.get('face', 'N/A')}")
    lines.append(f"**Chemical Formula**: {struct_info.get('formula', 'N/A')}")
    lines.append(f"**Number of Atoms**: {struct_info.get('num_atoms', 'N/A')}")
    lines.append("")

    if state.file_paths.get("unrelaxed_xyz"):
        lines.append(f"**Structure File**: `{Path(state.file_paths['unrelaxed_xyz']).name}`")
        lines.append("")

    return "\n".join(lines)


def generate_relaxation_section(state: "WorkflowState") -> Optional[str]:  # noqa: F821
    """Generate detailed relaxation section."""
    if not state.relaxation_results:
        return None

    lines = []
    lines.append("## Structure Relaxation")
    lines.append("")

    relax = state.relaxation_results
    lines.append(f"**Initial Energy**: {relax.get('initial_energy', 0):.4f} eV")
    lines.append(f"**Final Energy**: {relax.get('final_energy', 0):.4f} eV")
    lines.append(f"**Total Change**: {relax.get('energy_change', 0):.4f} eV")
    lines.append("")

    if state.file_paths.get("relaxed_xyz"):
        lines.append(f"**Relaxed Structure File**: `{Path(state.file_paths['relaxed_xyz']).name}`")
        lines.append("")

    if state.file_paths.get("visualization"):
        lines.append(f"**Visualization**: `{Path(state.file_paths['visualization']).name}`")
        lines.append("")

    return "\n".join(lines)


def generate_microscopy_section(state: "WorkflowState") -> Optional[str]:  # noqa: F821
    """Generate detailed microscopy section."""
    if not state.microscopy_results:
        return None

    lines = []
    lines.append("## Microscopy Simulations")
    lines.append("")

    for micro_type, results in state.microscopy_results.items():
        if not isinstance(results, dict) or not results:
            continue

        type_name = micro_type.upper()
        lines.append(f"### {type_name} Simulation")
        lines.append("")

        # Add results
        for key, value in results.items():
            if key == "results_file":
                lines.append(f"**Results File**: `{Path(value).name}`")
            elif key == "auxmaps_file":
                lines.append(f"**Auxiliary Maps**: `{Path(value).name}`")
            elif key == "parameters_file":
                lines.append(f"**Parameters File**: `{Path(value).name}`")
            elif key not in ["status", "method", "note", "error"]:
                if isinstance(value, (int, float)):
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                elif isinstance(value, str):
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        lines.append("")

    return "\n".join(lines)


def generate_full_report(
    state: "WorkflowState",  # noqa: F821
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate complete markdown report.

    Args:
        state: WorkflowState object with all workflow information
        output_dir: Directory to save the report (uses structure_dir if not provided)

    Returns:
        Complete markdown report as string
    """
    lines = []

    # Header
    element = state.structure_info.get("element", "Structure") if state.structure_info else "Workflow"
    face = state.structure_info.get("face", "") if state.structure_info else ""

    if element and face:
        title = f"Workflow Report: {element}({face})"
    else:
        title = "Workflow Report"

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Summary Section
    lines.append("## Summary")
    lines.append("")
    summary = generate_task_summary(state)
    # Remove bullet points from summary
    summary = summary.replace("- **", "**").replace("✓ ", "✓ ")
    lines.append(summary)
    lines.append("")

    # ... existing sections ...

    # Metadata
    lines.append("## Workflow Information")
    lines.append("")
    lines.append(f"**Session ID**: `{state.session_id}`")
    lines.append(f"**Started**: {state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Final Stage**: {state.workflow_stage}")
    lines.append(f"**AI Agent**: {detect_ai_agent(state.parsed_params)}")
    lines.append("")

    # Errors/Warnings Summary
    if state.errors or state.warnings:
        lines.append("## Issues")
        lines.append("")

        if state.errors:
            lines.append(f"### Errors ({len(state.errors)})")
            lines.append("")
            for error in state.errors:
                lines.append(f"- {error}")
            lines.append("")

        if state.warnings:
            lines.append(f"### Warnings ({len(state.warnings)})")
            lines.append("")
            for warning in state.warnings:
                lines.append(f"- {warning}")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by µ-Stack Workflow Engine*")

    report = "\n".join(lines)

    # Save report if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "workflow_report.md"

        try:
            with open(report_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return report


if __name__ == "__main__":
    print("Report generator module loaded successfully")
    print("\nAvailable functions:")
    print(" - detect_ai_agent(parsed_params)")
    print(" - generate_task_summary(state)")
    print(" - generate_structure_section(state)")
    print(" - generate_relaxation_section(state)")
    print(" - generate_microscopy_section(state)")
    print(" - generate_full_report(state, output_dir)")
