"""Interactive chat interface for ATOMIC."""

import os
import sys
import warnings
import uuid
from typing import Optional, Dict, Any

# Suppress warnings before any imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

from atomic_materials.utils import config
from atomic_materials.utils.logging import get_logger

console = Console()
logger = get_logger("interactive")

task_id = str(uuid.uuid4())[:8]


def print_logo():
    """Print ATOMIC ASCII logo."""
    logo = """[blue]      .       .       *       .       .[/blue]
[blue]    *   .   *   .   *   .   *   .[/blue]     [cyan]   ___   __________  __  ___________[/cyan]
[blue]      .   .   .   .   .   .   .[/blue]       [cyan]  / _ | /_  __/ __ \\/  |/  /  _/ ___/[/cyan]
[blue]    .   *   .   *   .   *   .   *[/blue]     [cyan] / __ |  / / / /_/ / /|_/ // // /__  [/cyan]
[blue]      .   .   .   .   .   .   .[/blue]       [cyan]/_/ |_| /_/  \\____/_/  /_/___/\\___/  [/cyan]
[blue]    AI MATERIALS SCIENTIST[/blue]
    """
    console.print(logo)


def parse_user_input(user_input: str) -> Dict[str, Any]:
    """Parse user input to extract parameters.

    This handles both relaxation and microscopy queries.
    """
    words = user_input.lower().split()

    known_elements = config.SUPPORTED_METALS + [e.lower() for e in config.SUPPORTED_2D]
    known_faces = config.SUPPORTED_FACES
    microscopy_types = ["tem", "afm", "stm", "iets", "ters"]

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
    from atomic_materials.relaxation.generate_surfaces import create_surface
    from atomic_materials.relaxation.surface_relaxation import (
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
    from atomic_materials.relaxation.generate_surfaces import create_surface
    from atomic_materials.relaxation.surface_relaxation import (
        load_model,
        relax_surfaces,
        plot_surface_relaxation,
    )
    from atomic_materials.relaxation.comparison import full_analysis
    from atomic_materials.relaxation.relax_report_generator import generate_full_report
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

    console.print(
        f"[bold green]✓ Analysis complete! Report saved to {report_file.name}[/bold green]"
    )

    return {
        "atoms": relaxed,
        "num_atoms": len(relaxed),
        "unrelaxed_file": str(unrelaxed_file),
        "relaxed_file": str(relaxed_file),
        "visualization": str(figure_file),
        "report_file": str(report_file),
        "initial_energy": init_e,
        "final_energy": final_e,
        "energy_change": final_e - init_e,
        "analysis": analysis,
    }


def run_interactive():
    """Run the interactive chat interface."""
    from atomic_materials.utils.gpu_detection import (
        get_torch_device,
        get_gpu_memory_info,
    )

    print_logo()

    console.print("[bold]Welcome to ATOMIC Interactive Mode![/bold]")
    console.print()

    # Show configured LLM agent
    llm_status = f"[cyan]{config.LLM_AGENT.upper()}[/cyan]"
    if config.LLM_AGENT == "anthropic":
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
    console.print("I can analyze atomic surfaces using ML potentials (MACE-MP).")
    console.print()
    console.print("[yellow]Try:[/yellow]")
    console.print(
        "  • [cyan]analyze Cu 100[/cyan] - Full analysis with AI-generated report"
    )
    console.print("  • [cyan]relax Pt 111[/cyan] - Quick relaxation without report")
    console.print("  • [cyan]generate graphene[/cyan] - Just create the structure")
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

            console.print("[dim]Processing...[/dim]")

            # Parse input
            params = parse_user_input(user_input)

            # If action is not determined or requires LLM
            if not params["element"] or params["use_llm"]:
                if config.LLM_AGENT in ["anthropic", "deepseek"]:
                    console.print(
                        f"[yellow]Using {config.LLM_AGENT.upper()} to parse query...[/yellow]"
                    )
                    console.print(
                        "[dim]LLM parsing not yet integrated in interactive mode[/dim]"
                    )
                    console.print(
                        "[dim]Please use direct commands like 'relax Cu 100'[/dim]\n"
                    )
                    continue
                else:
                    console.print("[yellow]⚠ I didn't understand the query.[/yellow]")
                    console.print(
                        "[dim]Try: 'relax Cu 100' or 'generate Pt 111'[/dim]\n"
                    )
                    continue

            # Show parsed parameters
            show_parameters(params)

            # Ask for confirmation
            proceed = Prompt.ask(
                "[yellow]Proceed?[/yellow]",
                choices=["yes", "no", "y", "n"],
                default="yes",
            )

            if proceed.lower() not in ["yes", "y"]:
                console.print("[dim]Cancelled.[/dim]\n")
                continue

            # Execute workflow
            console.print()
            try:
                if params["action"] in ["relax", "generate"]:
                    results = run_relaxation_workflow(
                        params["element"], params["face"], params["relax"]
                    )

                    # Show results
                    console.print("\n[bold green]✓ Workflow complete![/bold green]\n")

                    # Results table
                    results_table = Table(
                        title="Results", show_header=True, header_style="bold magenta"
                    )
                    results_table.add_column("Property", style="cyan", width=25)
                    results_table.add_column("Value", style="green")

                    results_table.add_row("Number of Atoms", str(results["num_atoms"]))

                    if params["relax"] and results["energy_change"]:
                        results_table.add_row(
                            "Initial Energy", f"{results['initial_energy']:.4f} eV"
                        )
                        results_table.add_row(
                            "Final Energy", f"{results['final_energy']:.4f} eV"
                        )
                        results_table.add_row(
                            "Energy Change", f"{results['energy_change']:.4f} eV"
                        )

                    console.print(results_table)

                    # Output files
                    console.print("\n[bold]Output Files:[/bold]")
                    console.print(f"  [green]✓[/green] {results['unrelaxed_file']}")
                    if results["relaxed_file"]:
                        console.print(f"  [green]✓[/green] {results['relaxed_file']}")
                        console.print(f"  [green]✓[/green] {results['visualization']}")

                    console.print()

                elif params["action"] == "analyze":
                    # Full analysis with report generation
                    results = run_analysis_workflow(params["element"], params["face"])

                    # Show results
                    console.print("\n[bold green]✓ Analysis complete![/bold green]\n")

                    # Results table
                    results_table = Table(
                        title="Results", show_header=True, header_style="bold magenta"
                    )
                    results_table.add_column("Property", style="cyan", width=25)
                    results_table.add_column("Value", style="green")

                    results_table.add_row("Number of Atoms", str(results["num_atoms"]))
                    results_table.add_row(
                        "Initial Energy", f"{results['initial_energy']:.4f} eV"
                    )
                    results_table.add_row(
                        "Final Energy", f"{results['final_energy']:.4f} eV"
                    )
                    results_table.add_row(
                        "Energy Change", f"{results['energy_change']:.4f} eV"
                    )

                    console.print(results_table)

                    # Output files
                    console.print("\n[bold]Output Files:[/bold]")
                    console.print(f"  [green]✓[/green] {results['unrelaxed_file']}")
                    console.print(f"  [green]✓[/green] {results['relaxed_file']}")
                    console.print(f"  [green]✓[/green] {results['visualization']}")
                    console.print(f"  [green]✓[/green] {results['report_file']}")

                    console.print()

                elif params["action"] == "microscopy":
                    console.print(
                        f"[yellow]Microscopy simulation ({params['microscopy_type']}) will be implemented soon.[/yellow]"
                    )
                    console.print(
                        "[dim]Surface relaxation completed. Microscopy simulation pending.[/dim]\n"
                    )

            except Exception as e:
                # Print error without markup to avoid Rich parsing issues
                console.print("\n[bold red]✗ Error:[/bold red]")
                console.print(str(e), style="red", markup=False)
                console.print()
                logger.error(f"Workflow failed: {e}", exc_info=True)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Quitting ATOMIC. Goodbye![/yellow]\n")
            sys.exit(0)
        except Exception as e:
            # Print error without markup to avoid Rich parsing issues
            console.print("\n[red]Error:[/red]")
            console.print(str(e), style="red", markup=False)
            console.print()


if __name__ == "__main__":
    run_interactive()
