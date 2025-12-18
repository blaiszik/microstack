"""Interactive chat interface for MicroStack."""

import os
import sys
import warnings
import logging
import uuid
from typing import Optional, Dict, Any

# Suppress warnings before any imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Suppress verbose logging from external packages
logging.getLogger("scilink").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("edison_client").setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.style import Style

from microstack.utils import config
from microstack.utils.logging import get_logger

console = Console()
logger = get_logger("interactive")

task_id = str(uuid.uuid4())[:8]


def interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    """Interpolate between two RGB colors. t=0 returns color1, t=1 returns color2."""
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))


def apply_gradient_to_text(text: str, start_color: tuple, end_color: tuple) -> str:
    """Apply a smooth RGB gradient to text."""
    result = ""
    for i, char in enumerate(text):
        progress = i / max(len(text) - 1, 1)
        color = interpolate_color(start_color, end_color, progress)
        r, g, b = color
        result += f"[rgb({r},{g},{b})]{char}[/rgb({r},{g},{b})]"
    return result


def print_logo():
    """Print MicroStack ASCII logo with smooth RGB gradient colors."""
    # Define gradient colors: cyan to magenta
    start_color = (0, 255, 255)  # Cyan
    end_color = (255, 0, 255)  # Magenta

    # STACK text lines
    stack_lines = [
        "███████╗████████╗ █████╗  ██████╗██╗  ██╗",
        "██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝",
        "███████╗   ██║   ███████║██║     █████╔╝ ",
        "╚════██║   ██║   ██╔══██║██║     ██╔═██╗ ",
        "███████║   ██║   ██║  ██║╚██████╗██║  ██╗",
        "╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝",
    ]

    # Apply gradient to each line
    gradient_lines = [
        apply_gradient_to_text(line, start_color, end_color) for line in stack_lines
    ]

    # Left side ASCII art (cyan)
    left_lines = [
        "[bold cyan]       ***           [/bold cyan]",
        "[bold cyan]       ***           [/bold cyan]",
        "[bold cyan]      ****     **    [/bold cyan]",
        "[bold cyan]      ****     **    [/bold cyan]",
        "[bold cyan]      ****     **    [/bold cyan]",
        "[bold cyan]     *****    ***    [/bold cyan]",
        "[bold cyan]     *  *********  * [/bold cyan]",
        "[bold cyan]    **   *****  ***  [/bold cyan]",
        "[bold cyan]   **      *         [/bold cyan]",
        "[bold cyan] ****                [/bold cyan]",
        "[bold cyan]****                 [/bold cyan]",
        "[bold cyan]***                  [/bold cyan]",
    ]

    # Print logo line by line
    console.print(left_lines[0])
    console.print(left_lines[1])
    console.print(left_lines[2])
    console.print(left_lines[3] + "       " + gradient_lines[0])
    console.print(left_lines[4] + "       " + gradient_lines[1])
    console.print(left_lines[5] + "[bold cyan]█████╗ [/bold cyan]" + gradient_lines[2])
    console.print(left_lines[6] + "[bold cyan]╚════╝ [/bold cyan]" + gradient_lines[3])
    console.print(left_lines[7] + "       " + gradient_lines[4])
    console.print(left_lines[8] + "       " + gradient_lines[5])
    console.print(left_lines[9])
    console.print(left_lines[10])
    console.print(left_lines[11])
    console.print()


def parse_user_input(user_input: str) -> Dict[str, Any]:
    """Parse user input to extract parameters.

    This handles both relaxation and microscopy queries.
    """
    words = user_input.lower().split()

    known_elements = config.SUPPORTED_METALS + [e.lower() for e in config.SUPPORTED_2D]
    known_faces = config.SUPPORTED_FACES
    microscopy_types = ["afm", "stm", "iets"]

    params = {
        "action": None,  # "relax", "generate", "microscopy", "analyze"
        "element": None,
        "face": None,
        "microscopy_type": None,
        "relax": False,
        "use_llm": False,  # Whether to use LLM for complex parsing
    }

    # Check for microscopy types
    for word in words:
        if word in microscopy_types:
            params["microscopy_type"] = word.upper()
            params["action"] = "microscopy"
            params["use_llm"] = True  # Complex query, use LLM
            break

    # Check for action keywords
    if "analyze" in words or "analysis" in words or "report" in words:
        params["action"] = "analyze"
    elif "relax" in words:
        params["action"] = "relax"
        params["relax"] = True
    elif "generate" in words or "create" in words:
        params["action"] = "generate"
        params["relax"] = False

    # Find element
    for word in words:
        if word == "graphene":
            params["element"] = "C"
            params["face"] = "graphene"
            break

        word_upper = word.upper()
        if word_upper in ["MOS2", "WS2", "MOSE2", "WSE2"]:
            if word_upper == "MOS2":
                params["element"] = "MoS2"
            elif word_upper == "WS2":
                params["element"] = "WS2"
            elif word_upper == "MOSE2":
                params["element"] = "MoSe2"
            elif word_upper == "WSE2":
                params["element"] = "WSe2"
            params["face"] = "2d"
            continue

        w_cap = word.capitalize()
        if w_cap in config.SUPPORTED_METALS or w_cap in [
            e.capitalize() for e in config.SUPPORTED_2D
        ]:
            params["element"] = w_cap

        if word in known_faces:
            params["face"] = word

    # Default face if not specified
    if params["element"] and not params["face"]:
        if params["element"] == "C":
            params["face"] = "graphene"
        elif params["element"] in config.SUPPORTED_2D:
            params["face"] = "2d"
        else:
            params["face"] = "100"

    return params


def show_parameters(params: Dict[str, Any], microscopy_info: Optional[Dict] = None):
    """Display parsed parameters in a nice table."""
    table = Table(
        title="[bold cyan]Parsed Parameters[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green")

    # Action
    if params["action"]:
        table.add_row("Action", params["action"].capitalize())

    # Surface parameters
    if params["element"]:
        table.add_row("Element", str(params["element"]))
    if params["face"]:
        table.add_row("Surface Face", str(params["face"]))

    # Relaxation
    if params["relax"]:
        table.add_row("Relaxation", "Yes")
        table.add_row("Relaxation Steps", str(config.DEFAULT_RELAXATION_STEPS))
    else:
        table.add_row("Relaxation", "No")

    # Microscopy info (if parsed by LLM)
    if params["microscopy_type"]:
        table.add_row("Microscopy Type", params["microscopy_type"])

    if microscopy_info:
        for key, value in microscopy_info.items():
            if value is not None:
                table.add_row(key.replace("_", " ").title(), str(value))

    console.print()
    console.print(table)
    console.print()


def run_relaxation_workflow(element: str, face: str, relax: bool) -> Dict[str, Any]:
    """Run surface generation and optional relaxation."""
    from microstack.relaxation.generate_surfaces import create_surface
    from microstack.relaxation.surface_relaxation import (
        load_model,
        relax_surfaces,
        plot_surface_relaxation,
    )
    from ase.io import write

    # Initialize output directory
    config.init_output_dirs()

    console.print(f"\n[cyan]Generating {element}({face}) surface...[/cyan]")

    # Generate surface
    atoms, task_dir = create_surface(element, face, task_id)
    console.print(f"[green]✓[/green] Created surface with {len(atoms)} atoms")

    # Save unrelaxed structure
    unrelaxed_file = task_dir / f"{element}_{face}_unrelaxed.xyz"
    write(str(unrelaxed_file), atoms)

    results = {
        "atoms": atoms,
        "num_atoms": len(atoms),
        "unrelaxed_file": str(unrelaxed_file),
        "relaxed_file": None,
        "initial_energy": None,
        "final_energy": None,
        "energy_change": None,
    }

    if relax:
        console.print(f"\n[cyan]Loading MACE model...[/cyan]")
        model = load_model()

        console.print(
            f"[cyan]Relaxing surface ({config.DEFAULT_RELAXATION_STEPS} steps)...[/cyan]"
        )
        relaxed_surfaces, initial_energies, final_energies = relax_surfaces(
            [atoms], model, steps=config.DEFAULT_RELAXATION_STEPS
        )

        relaxed_atoms = relaxed_surfaces[0]
        init_e = initial_energies[0]
        final_e = final_energies[0]

        console.print(f"[green]✓[/green] Relaxation complete")
        console.print(
            f"  Energy: {init_e:.4f} → {final_e:.4f} eV (Δ = {final_e - init_e:.4f} eV)"
        )

        # Save relaxed structure
        relaxed_file = task_dir / f"{element}_{face}_relaxed.xyz"
        write(str(relaxed_file), relaxed_atoms)

        # Generate visualization
        viz_file = task_dir / f"{element}_{face}_relaxation.png"
        plot_surface_relaxation(
            [atoms], [relaxed_atoms], [f"{element}({face})"], filename=str(viz_file)
        )

        results.update(
            {
                "relaxed_file": str(relaxed_file),
                "initial_energy": init_e,
                "final_energy": final_e,
                "energy_change": final_e - init_e,
                "visualization": str(viz_file),
            }
        )

    return results


def run_analysis_workflow(element: str, face: str) -> Dict[str, Any]:
    """Run full analysis workflow with report generation."""
    from microstack.relaxation.generate_surfaces import create_surface
    from microstack.relaxation.surface_relaxation import (
        load_model,
        relax_surfaces,
        plot_surface_relaxation,
    )
    from microstack.relaxation.comparison import full_analysis
    from microstack.relaxation.relax_report_generator import (
        generate_full_report,
        generate_natural_description,
    )
    from ase.io import write

    # Initialize output directory
    config.init_output_dirs()

    console.print(f"\n[cyan][1/5] Generating {element}({face}) surface...[/cyan]")
    atoms, task_dir = create_surface(element, face, task_id)
    unrelaxed = atoms.copy()
    console.print(f"      Created {element}({face}) with {len(atoms)} atoms")

    # Save unrelaxed structure
    unrelaxed_file = task_dir / f"{element}_{face}_unrelaxed.xyz"
    write(str(unrelaxed_file), atoms)

    console.print(f"\n[cyan][2/5] Loading MACE model...[/cyan]")
    model = load_model()

    console.print(
        f"[cyan][3/5] Relaxing surface ({config.DEFAULT_RELAXATION_STEPS} steps)...[/cyan]"
    )
    relaxed_surfaces, initial_energies, final_energies = relax_surfaces(
        [atoms], model, steps=config.DEFAULT_RELAXATION_STEPS
    )

    relaxed = relaxed_surfaces[0]
    init_e = initial_energies[0]
    final_e = final_energies[0]

    console.print(
        f"      Energy: {init_e:.4f} → {final_e:.4f} eV (Δ = {final_e - init_e:.4f} eV)"
    )

    # Save relaxed structure
    relaxed_file = task_dir / f"{element}_{face}_relaxed.xyz"
    write(str(relaxed_file), relaxed)

    console.print(f"\n[cyan][4/5] Generating visualization...[/cyan]")
    figure_file = task_dir / f"{element}_{face}_relaxation.png"
    plot_surface_relaxation(
        [unrelaxed], [relaxed], [f"{element}({face})"], filename=str(figure_file)
    )

    console.print(f"[cyan][5/5] Analyzing and generating report...[/cyan]")
    analysis = full_analysis(
        unrelaxed=unrelaxed,
        relaxed=relaxed,
        element=element,
        face=face,
        initial_energy=init_e,
        final_energy=final_e,
    )

    # Check for reference data
    if analysis["comparison"]["has_reference"]:
        console.print(
            f"      Comparing with: {analysis['comparison']['reference_source']}"
        )
        console.print(
            f"      Agreement: [green]{analysis['comparison']['overall_agreement'].upper()}[/green]"
        )
    else:
        console.print("      No reference data available for comparison")

    # Generate report with AI discussion
    console.print("      Generating AI discussion...")
    report = generate_full_report(
        element=element, face=face, analysis=analysis, figure_paths=[str(figure_file)]
    )

    # Save report
    report_file = task_dir / f"{element}_{face}_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    # Generate and display Claude Sonnet 4.5 natural language summary
    console.print("      Generating natural language summary (Claude Sonnet 4.5)...")
    summary = generate_natural_description(element, face, analysis)

    # Save summary as separate markdown file
    summary_file = task_dir / f"{element}_{face}_summary.md"
    with open(summary_file, "w") as f:
        f.write(f"# {element}({face}) Surface Relaxation Summary\n\n")
        f.write(summary)
        f.write("\n\n---\n*Generated by MicroStack using Claude Sonnet 4.5*\n")

    console.print(f"[bold green]✓ Analysis complete![/bold green]")

    # Display the summary in a nice panel
    console.print()
    console.print(
        Panel(
            summary,
            title="[bold cyan]AI Summary (Claude Sonnet 4.5)[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    return {
        "atoms": relaxed,
        "num_atoms": len(relaxed),
        "unrelaxed_file": str(unrelaxed_file),
        "relaxed_file": str(relaxed_file),
        "visualization": str(figure_file),
        "report_file": str(report_file),
        "summary_file": str(summary_file),
        "summary": summary,
        "initial_energy": init_e,
        "final_energy": final_e,
        "energy_change": final_e - init_e,
        "analysis": analysis,
    }


def _display_workflow_results(final_state):
    """Display results from completed workflow."""
    # Check for errors
    if final_state.has_errors():
        console.print("[bold red]✗ Workflow encountered errors:[/bold red]")
        for error in final_state.errors:
            console.print(f"  [red]• {error}[/red]")
        return

    # Show success
    console.print("[bold green]✓ Workflow completed successfully![/bold green]")
    console.print()

    # Summary table
    summary_table = Table(
        title="[cyan]Workflow Summary[/cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Property", style="cyan", width=25)
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Session ID", final_state.session_id)
    summary_table.add_row("Stage", final_state.workflow_stage)

    if final_state.structure_info:
        summary_table.add_row(
            "Formula", final_state.structure_info.get("formula", "N/A")
        )
        summary_table.add_row(
            "Atoms", str(final_state.structure_info.get("num_atoms", "N/A"))
        )

    if final_state.relaxation_results:
        init_e = final_state.relaxation_results.get("initial_energy")
        final_e = final_state.relaxation_results.get("final_energy")
        if init_e and final_e:
            summary_table.add_row("Initial Energy", f"{init_e:.4f} eV")
            summary_table.add_row("Final Energy", f"{final_e:.4f} eV")
            summary_table.add_row("Energy Change", f"{final_e - init_e:.4f} eV")

    if final_state.microscopy_type:
        # Handle both single microscopy type (string) and multiple types (list)
        if isinstance(final_state.microscopy_type, list):
            microscopy_display = " → ".join(final_state.microscopy_type)
        else:
            microscopy_display = final_state.microscopy_type
        summary_table.add_row("Microscopy Type", microscopy_display)

    console.print(summary_table)
    console.print()

    # Output files
    if final_state.file_paths:
        console.print("[bold]Output Files:[/bold]")
        for key, path in final_state.file_paths.items():
            if path and key != "output_dir":
                console.print(f"  [green]✓[/green] {key}: {path}")

    # Warnings
    if final_state.warnings:
        console.print()
        console.print("[yellow]⚠ Warnings:[/yellow]")
        for warning in final_state.warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")

    # Microscopy results
    if final_state.microscopy_results:
        console.print()
        console.print("[bold cyan]Microscopy Results:[/bold cyan]")
        for microscopy_type, results in final_state.microscopy_results.items():
            console.print(f"  [cyan]{microscopy_type.upper()}:[/cyan]")
            for key, value in results.items():
                if key.endswith("_file") or key.endswith("_dir"):
                    console.print(f"    {key}: {value}")
                else:
                    console.print(f"    {key}: {value}")

    # Generate comprehensive workflow report
    try:
        from microstack.utils.report_generator import (
            generate_task_summary,
            generate_full_report,
        )
        from pathlib import Path

        console.print()
        console.print("[cyan]Generating workflow report...[/cyan]")

        # Get the structure directory (parent of relaxation directory)
        structure_dir = None
        if final_state.file_paths and final_state.file_paths.get("structure_dir"):
            structure_dir = Path(final_state.file_paths["structure_dir"])
        elif final_state.file_paths and final_state.file_paths.get("output_dir"):
            structure_dir = Path(final_state.file_paths["output_dir"]).parent

        # Display summary in terminal
        summary = generate_task_summary(final_state)
        console.print()
        console.print("[bold cyan]Task Summary[/bold cyan]")
        console.print(summary)

        # Generate and save full report to structure directory
        if structure_dir:
            full_report = generate_full_report(final_state, structure_dir)
            console.print(f"[green]✓[/green] Full report saved to {structure_dir}")
        else:
            full_report = generate_full_report(final_state)
            console.print("[yellow]⚠[/yellow] Could not determine structure directory for report")

    except Exception as e:
        logger.warning(f"Failed to generate workflow report: {e}")

    # Generate and display AI summary if we have relaxation results
    if final_state.relaxation_results and final_state.structure_info:
        try:
            from microstack.relaxation.relax_report_generator import (
                generate_natural_description,
            )
            from microstack.utils.report_generator import detect_ai_agent
            from pathlib import Path

            # Detect which AI agent is being used
            ai_agent = detect_ai_agent(final_state.parsed_params)

            console.print()
            console.print(
                f"[cyan]Generating AI summary ({ai_agent})...[/cyan]"
            )

            # Build analysis dict from workflow state
            formula = final_state.structure_info.get("formula", "Unknown")
            analysis = {
                "energy_change_eV": (
                    final_state.relaxation_results.get("final_energy", 0)
                    - final_state.relaxation_results.get("initial_energy", 0)
                ),
                "relaxation": {
                    "max_displacement": final_state.relaxation_results.get(
                        "max_displacement", 0
                    ),
                    "n_atoms": final_state.structure_info.get("num_atoms", 0),
                    "layer_changes_percent": final_state.relaxation_results.get(
                        "layer_changes", {}
                    ),
                },
                "comparison": {
                    "overall_agreement": "not available",
                },
                "microscopy_results": final_state.microscopy_results or {},
            }

            # Generate the summary
            summary = generate_natural_description(formula, "surface", analysis)

            # Save summary to base structure directory (not relaxation subdirectory)
            if final_state.file_paths:
                # Get structure_dir (base directory) or derive it from output_dir
                structure_dir = final_state.file_paths.get("structure_dir")
                if not structure_dir and final_state.file_paths.get("output_dir"):
                    structure_dir = str(
                        Path(final_state.file_paths["output_dir"]).parent
                    )

                if structure_dir:
                    structure_dir = Path(structure_dir)
                    structure_dir.mkdir(parents=True, exist_ok=True)
                    summary_file = structure_dir / f"{formula}_ai_summary.md"
                    with open(summary_file, "w") as f:
                        f.write(f"# {formula} Surface Relaxation Summary\n\n")
                        f.write(summary)
                        f.write(
                            f"\n\n---\n*Generated by MicroStack using {ai_agent}*\n"
                        )
                    console.print(
                        f"  [green]✓[/green] ai_summary_file: {summary_file}"
                    )

            # Display the summary
            console.print()
            console.print(
                Panel(
                    summary,
                    title=f"[bold cyan]AI Summary ({ai_agent})[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        except Exception as e:
            console.print(f"[yellow]⚠ Could not generate AI summary: {e}[/yellow]")


def run_interactive():
    """Run the interactive chat interface."""
    from microstack.utils.gpu_detection import (
        get_torch_device,
        get_gpu_memory_info,
    )
    from microstack.agents.workflow import run_workflow

    print_logo()

    console.print("[bold]Welcome to MicroStack Interactive Mode![/bold]")
    console.print()

    # Show configured LLM agent
    llm_status = f"[cyan]{config.LLM_AGENT.upper()}[/cyan]"
    if config.LLM_AGENT == "gemini":
        if config.GOOGLE_API_KEY:
            llm_status += " [green]✓[/green]"
        else:
            llm_status += " [red]✗[/red]"
    elif config.LLM_AGENT == "anthropic":
        if config.ANTHROPIC_API_KEY:
            llm_status += " [green]✓[/green]"
        else:
            llm_status += " [red]✗[/red]"
    elif config.LLM_AGENT == "deepseek":
        if config.DEEPSEEK_API_KEY:
            llm_status += " [green]✓[/green]"
        else:
            llm_status += " [red]✗[/red]"

    console.print(f"LLM Agent: {llm_status}")

    # Show device and GPU info
    device = get_torch_device()
    console.print(f"[cyan]Device:[/cyan] {device}")

    if "cuda" in str(device):
        gpu_mem = get_gpu_memory_info("cuda")
        console.print(
            f"[cyan]GPU Memory:[/cyan] {gpu_mem['total_gb']:.1f} GB (Free: {gpu_mem['free_gb']:.1f} GB)"
        )
    else:
        console.print(f"[cyan]GPU Memory:[/cyan] CPU mode")

    console.print()
    console.print(
        "I can generate atomic structures and analyze them with microscopy simulations!"
    )
    console.print()
    console.print("[yellow]Try:[/yellow]")
    console.print(
        "  • [cyan]Build a 3x3x4 Cu(111) surface with 15A vacuum[/cyan] - SciLink structure generation"
    )
    console.print(
        "  • [cyan]Generate graphene (001) with 10A vacuum, then STM[/cyan] - Structure + microscopy"
    )
    console.print(
        "  • [cyan]Relax Cu 100[/cyan] - Simple surface generation and relaxation"
    )
    console.print()
    console.print("[dim]Type 'quit' or 'exit' to leave.[/dim]")
    console.print()

    # Check config
    warnings_list = config.validate_config()
    if warnings_list:
        console.print("[yellow]⚠ Configuration warnings:[/yellow]")
        for w in warnings_list:
            console.print(f"  {w}")
        console.print()

    while True:
        try:
            user_input = Prompt.ask("[green]You[/green]")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]\n")
                break

            if not user_input.strip():
                continue

            # Create session
            session_id = str(uuid.uuid4())[:8]

            with console.status(
                "[yellow]Processing query with LLM...[/yellow]", spinner="dots"
            ):
                # Run the LangGraph workflow
                final_state = run_workflow(user_input, session_id)

            # Display results
            console.print()
            _display_workflow_results(final_state)
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Quitting MicroStack. Goodbye![/yellow]\n")
            sys.exit(0)
        except Exception as e:
            # Print error without markup to avoid Rich parsing issues
            console.print("\n[red]Error:[/red]")
            console.print(str(e), style="red")
            console.print()


if __name__ == "__main__":
    run_interactive()
