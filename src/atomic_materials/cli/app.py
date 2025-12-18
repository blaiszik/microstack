"""ATOMIC CLI application - Main entry point."""

import os
import sys
import warnings
import logging
import uuid

# Suppress warnings before other imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Suppress verbose logging from external packages
logging.getLogger("scilink").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("edison_client").setLevel(logging.WARNING)

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from atomic_materials.utils import config
from atomic_materials.utils.logging import get_logger

console = Console()
logger = get_logger("cli")


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="0.1.0", prog_name="ATOMIC")
def cli(ctx: click.Context) -> None:
    """
    ATOMIC - AI Materials Scientist.

    Analyze atomic surfaces using Machine Learning Potentials,
    with experimental validation and AI-generated scientific reports.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, show welcome and enter interactive mode
        show_welcome()
        ctx.invoke(interactive)


@cli.command()
def interactive() -> None:
    """
    Start interactive chat mode for conversational simulation creation.

    Chat with ATOMIC to generate surfaces, relax structures, and run simulations.
    """
    from atomic_materials.cli.interactive import run_interactive

    run_interactive()


@cli.command()
@click.argument("element", type=str)
@click.argument("face", type=str)
@click.option(
    "--relax/--no-relax", default=True, help="Relax surface structure (default: yes)"
)
@click.option("--steps", default=None, type=int, help="Number of relaxation steps")
@click.option("--output-dir", type=click.Path(), help="Output directory")
def relax(element: str, face: str, relax: bool, steps: int, output_dir: str) -> None:
    """
    Generate and optionally relax a surface structure.

    Examples:
        atomic relax Cu 100
        atomic relax Pt 111 --no-relax
        atomic relax C graphene --steps 300
    """
    from atomic_materials.relaxation.generate_surfaces import create_surface
    from atomic_materials.relaxation.surface_relaxation import (
        load_model,
        relax_surfaces,
        plot_surface_relaxation,
    )
    from ase.io import write

    console.print(f"\n[bold cyan]Surface Relaxation Workflow[/bold cyan]\n")

    # Show parameters in a nice table
    param_table = Table(
        title="Parameters", show_header=True, header_style="bold magenta"
    )
    param_table.add_column("Parameter", style="cyan", width=20)
    param_table.add_column("Value", style="green")

    param_table.add_row("Element", element)
    param_table.add_row("Surface Face", face)
    param_table.add_row("Relaxation", "Yes" if relax else "No")
    if relax:
        relaxation_steps = steps if steps else config.DEFAULT_RELAXATION_STEPS
        param_table.add_row("Relaxation Steps", str(relaxation_steps))

    console.print(param_table)
    console.print()

    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())[:8]

        # Initialize output directory
        config.init_output_dirs()

        # Generate surface - create_surface handles file saving automatically
        with console.status("[yellow]Generating surface...[/yellow]", spinner="dots"):
            atoms, output_path = create_surface(element, face, task_id)

        # If custom output directory specified, copy files there
        if output_dir:
            custom_output = Path(output_dir)
            custom_output.mkdir(parents=True, exist_ok=True)
            # Copy the unrelaxed file to custom directory
            unrelaxed_file_src = output_path / f"{element}_{face}_unrelaxed.xyz"
            unrelaxed_file = custom_output / f"{element}_{face}_unrelaxed.xyz"
            if unrelaxed_file_src.exists():
                import shutil

                shutil.copy(str(unrelaxed_file_src), str(unrelaxed_file))
            output_path = custom_output
        else:
            unrelaxed_file = output_path / f"{element}_{face}_unrelaxed.xyz"

        console.print(f"[green]✓[/green] Generated surface with {len(atoms)} atoms")

        if relax:
            # Load model
            with console.status(
                "[yellow]Loading MACE model...[/yellow]", spinner="dots"
            ):
                model = load_model()

            # Relax surface
            relaxation_steps = steps if steps else config.DEFAULT_RELAXATION_STEPS
            with console.status(
                f"[yellow]Relaxing surface ({relaxation_steps} steps)...[/yellow]",
                spinner="dots",
            ):
                relaxed_surfaces, initial_energies, final_energies = relax_surfaces(
                    [atoms], model, steps=relaxation_steps
                )

            relaxed_atoms = relaxed_surfaces[0]
            init_e = initial_energies[0]
            final_e = final_energies[0]

            console.print(f"[green]✓[/green] Relaxation complete")

            # Save relaxed structure
            relaxed_file = output_path / f"{element}_{face}_relaxed.xyz"
            write(str(relaxed_file), relaxed_atoms)

            # Generate visualization
            viz_file = output_path / f"{element}_{face}_relaxation.png"
            plot_surface_relaxation(
                [atoms], [relaxed_atoms], [f"{element}({face})"], filename=str(viz_file)
            )

            # Display results
            console.print(
                "\n[bold green]✓ Workflow completed successfully![/bold green]\n"
            )

            # Show statistics
            stats_table = Table(
                title="Results", show_header=True, header_style="bold magenta"
            )
            stats_table.add_column("Property", style="cyan", width=25)
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Number of Atoms", str(len(atoms)))
            stats_table.add_row("Initial Energy", f"{init_e:.4f} eV")
            stats_table.add_row("Final Energy", f"{final_e:.4f} eV")
            stats_table.add_row("Energy Change", f"{final_e - init_e:.4f} eV")

            console.print(stats_table)

            # Show output files
            console.print("\n[bold]Output Files:[/bold]")
            console.print(f"  [green]✓[/green] {unrelaxed_file}")
            console.print(f"  [green]✓[/green] {relaxed_file}")
            console.print(f"  [green]✓[/green] {viz_file}")
        else:
            console.print("\n[bold green]✓ Surface generated![/bold green]\n")
            console.print(
                f"[bold]Output File:[/bold]\n  [green]✓[/green] {unrelaxed_file}"
            )

        console.print()

    except Exception as e:
        # Print error without markup to avoid Rich parsing issues
        console.print("\n[bold red]✗ Error:[/bold red]")
        console.print(str(e), style="red")
        console.print()
        logger.error(f"Relaxation workflow failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--relax/--no-relax", default=True, help="Relax surface structure (default: yes)"
)
def query(query: str, relax: bool) -> None:
    """
    Generate structure via SciLink and relax using MACE.

    Example: atomic query "3x3x4 Cu(111) surface with 15A vacuum"
    """
    import scilink as sl
    from pathlib import Path
    from ase.io import write
    from atomic_materials.relaxation.surface_relaxation import (
        load_model,
        relax_surfaces,
        plot_surface_relaxation,
    )

    console.print(f"\n[bold cyan]SciLink Agent Input:[/bold cyan] {query}\n")

    try:
        # 1. Setup Task ID and Directories (UUID integration)
        task_id = str(uuid.uuid4())[:8]
        config.init_output_dirs()

        # Determine output path using app logic
        output_base = Path(config.OUTPUT_SUBDIRS["relaxation"])
        output_path = output_base / f"query_{task_id}"
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. SciLink Structure Generation
        engine = sl.Interface("ase")
        with console.status(
            "[yellow]SciLink is architecting the atoms...[/yellow]", spinner="dots"
        ):
            # Prompt engineering to ensure variable 'atoms' is created
            prompt = (
                f"Using ASE build functions, {query}. Store result in 'atoms' variable."
            )
            result = engine.run(prompt)
            atoms = result.get("atoms")

        if not atoms:
            raise ValueError("SciLink failed to generate an ASE Atoms object.")

        # Save the initial unrelaxed structure
        formula = atoms.get_chemical_formula()
        unrelaxed_file = output_path / f"{formula}_unrelaxed.xyz"
        write(str(unrelaxed_file), atoms)
        console.print(f"[green]✓[/green] Generated {formula} with {len(atoms)} atoms")

        # 3. MACE Relaxation Pipeline
        if relax:
            with console.status(
                "[yellow]Loading MACE & Relaxing...[/yellow]", spinner="dots"
            ):
                model = load_model()
                # Batch size of 1
                relaxed_surfaces, init_e, final_e = relax_surfaces([atoms], model)
                relaxed_atoms = relaxed_surfaces[0]

            # Save Relaxed Structure
            relaxed_file = output_path / f"{formula}_relaxed.xyz"
            write(str(relaxed_file), relaxed_atoms)

            # Generate Visualization (Matching the relax command style)
            viz_file = output_path / f"{formula}_relaxation.png"
            plot_surface_relaxation(
                [atoms], [relaxed_atoms], [f"Query: {formula}"], filename=str(viz_file)
            )

            # 4. Success Summary Table
            res_table = Table(show_header=True, header_style="bold magenta")
            res_table.add_column("Property", style="cyan")
            res_table.add_column("Value", style="green")
            res_table.add_row("Task ID", task_id)
            res_table.add_row("Energy Change", f"{final_e[0] - init_e[0]:.4f} eV")
            res_table.add_row("Output Dir", str(output_path))

            console.print(res_table)
            console.print(f"\n[bold green]✓ Files saved to:[/bold green] {output_path}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Query Workflow Failed:[/bold red] {e}")
        logger.error(f"SciLink query failed: {e}", exc_info=True)


@cli.command()
@click.argument("query", type=str)
def simulate(query: str) -> None:
    """
    Run complete simulation workflow from natural language query.

    Uses LLM-powered parsing for structure generation and optional microscopy.

    Examples:
        atomic simulate "Build a 3x3x4 Cu(111) surface with 15A vacuum"
        atomic simulate "Generate Pt(111), relax it, then run STM"
        atomic simulate "Create graphene (001) with 10A vacuum and run AFM"
    """
    from atomic_materials.agents.workflow import run_workflow
    from rich.table import Table

    console.print(f"\n[bold cyan]ATOMIC Simulation Workflow[/bold cyan]\n")
    console.print(f"[cyan]Query:[/cyan] {query}\n")

    # Create session ID
    session_id = str(uuid.uuid4())[:8]

    try:
        # Run workflow
        with console.status(
            "[yellow]Running LangGraph workflow...[/yellow]", spinner="dots"
        ):
            final_state = run_workflow(query, session_id)

        # Display results
        console.print()

        # Check for errors
        if final_state.has_errors():
            console.print("[bold red]✗ Workflow completed with errors:[/bold red]")
            for error in final_state.errors:
                console.print(f"  [red]• {error}[/red]")
            console.print()

        # Success message
        if not final_state.has_errors():
            console.print("[bold green]✓ Workflow completed successfully![/bold green]")
        else:
            console.print(
                "[bold yellow]⚠ Workflow completed with warnings[/bold yellow]"
            )

        console.print()

        # Summary table
        summary_table = Table(
            title="[cyan]Simulation Summary[/cyan]",
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
            if init_e is not None and final_e is not None:
                summary_table.add_row("Initial Energy", f"{init_e:.4f} eV")
                summary_table.add_row("Final Energy", f"{final_e:.4f} eV")
                summary_table.add_row("Energy Change", f"{final_e - init_e:.4f} eV")

        if final_state.microscopy_type:
            summary_table.add_row("Microscopy Type", final_state.microscopy_type)

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

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Simulation Failed:[/bold red]")
        console.print(str(e), style="red")
        console.print()
        logger.error(f"Simulation workflow failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command("check-config")
def check_config() -> None:
    """
    Validate configuration and check API connectivity.
    """
    console.print("\n[bold cyan]Configuration Check[/bold cyan]\n")

    # Check LLM Agent
    console.print(f"[bold]LLM Agent:[/bold] {config.LLM_AGENT.upper()}")

    if config.LLM_AGENT == "anthropic":
        if config.ANTHROPIC_API_KEY:
            console.print("[green]✓[/green] Anthropic API key configured")
            try:
                client = config.get_anthropic_client()
                if client:
                    console.print(
                        "[green]✓[/green] Anthropic client initialized successfully"
                    )
            except Exception as e:
                console.print(f"[red]✗[/red] Anthropic client error: {e}")
        else:
            console.print("[red]✗[/red] Anthropic API key not set")

    elif config.LLM_AGENT == "deepseek":
        if config.DEEPSEEK_API_KEY:
            console.print("[green]✓[/green] DeepSeek API key configured")
            try:
                client = config.get_deepseek_client()
                if client:
                    console.print(
                        "[green]✓[/green] DeepSeek client initialized successfully"
                    )
            except Exception as e:
                console.print(f"[red]✗[/red] DeepSeek client error: {e}")
        else:
            console.print("[red]✗[/red] DeepSeek API key not set")

    # Check Materials Project API key
    console.print()
    if config.MATERIALS_PROJECT_API_KEY:
        console.print("[green]✓[/green] Materials Project API key configured")
    else:
        console.print("[yellow]⚠[/yellow] Materials Project API key not set (optional)")

    # Check GPU
    console.print("\n[bold]GPU Status:[/bold]")
    try:
        from atomic_materials.utils.gpu_detection import detect_gpu_capabilities

        gpu_caps = detect_gpu_capabilities()

        if gpu_caps["cuda_available"]:
            console.print(
                f"[green]✓[/green] CUDA: {gpu_caps['cuda_devices']} device(s)"
            )
            for i, name in enumerate(gpu_caps["cuda_device_names"]):
                console.print(f"  └─ Device {i}: {name}")
        else:
            console.print(
                "[yellow]⚠[/yellow] CUDA: Not available (CPU mode will be used)"
            )

        console.print(
            f"\n[bold]Recommended backend:[/bold] {gpu_caps['recommended_backend']}"
        )

    except Exception as e:
        console.print(f"[red]✗[/red] GPU detection error: {e}")

    # Check output directories
    console.print(f"\n[bold]Output Configuration:[/bold]")
    console.print(f"  • Base output directory: {config.OUTPUT_DIR}")
    console.print(f"  • Relaxation output: {config.OUTPUT_SUBDIRS['relaxation']}")

    # Show warnings
    warnings_list = config.validate_config()
    if warnings_list:
        console.print("\n[yellow]⚠ Configuration warnings:[/yellow]")
        for w in warnings_list:
            console.print(f"  {w}")

    console.print()


def show_welcome() -> None:
    """Display welcome message."""
    llm_agent = config.LLM_AGENT.upper()
    welcome_text = f"""[bold cyan]ATOMIC[/bold cyan] - AI Materials Scientist

Analyze atomic surfaces using Machine Learning Potentials!

[bold]Quick Start:[/bold]
  • [green]atomic[/green] - Start interactive mode (recommended)
  • [green]atomic relax Cu 100[/green] - Generate and relax Cu(100)
  • [green]atomic analyze Pt 111[/green] - Full analysis with AI report
  • [green]atomic check-config[/green] - Check configuration

[bold]LLM Agent:[/bold] {llm_agent} (Anthropic or DeepSeek)
[bold]Features:[/bold] Surface Generation | MACE ML Relaxation | AI Reports

[dim]Version 0.1.0 | GPU-Accelerated | Experimental Validation[/dim]"""

    panel = Panel(
        welcome_text,
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        # Print error without markup to avoid Rich parsing issues
        console.print("\n[red]Fatal error:[/red]")
        console.print(str(e), style="red")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
