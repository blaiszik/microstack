"""ATOMIC CLI application - Main entry point."""

import os
import sys
import warnings
import uuid

# Suppress warnings before other imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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
        console.print(str(e), style="red", markup=False)
        console.print()
        logger.error(f"Relaxation workflow failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
def query(query: str) -> None:
    """
    Parse a natural language query for microscopy simulation.

    This command uses the configured LLM agent (Anthropic or DeepSeek) to
    parse and understand microscopy simulation requests.

    Examples:
        atomic query "Generate TEM image for Si with 300 kV"
        atomic query "AFM scan of NaCl surface"
        atomic query "STM image of graphene"
    """
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    try:
        # Check LLM configuration
        if config.LLM_AGENT == "anthropic" and not config.ANTHROPIC_API_KEY:
            console.print("[bold red]✗ Anthropic API key not configured[/bold red]")
            console.print(
                "Set ANTHROPIC_API_KEY environment variable or in config.py\n"
            )
            sys.exit(1)
        elif config.LLM_AGENT == "deepseek" and not config.DEEPSEEK_API_KEY:
            console.print("[bold red]✗ DeepSeek API key not configured[/bold red]")
            console.print("Set DEEPSEEK_API_KEY environment variable or in config.py\n")
            sys.exit(1)

        console.print(
            f"[yellow]Parsing query with {config.LLM_AGENT.upper()}...[/yellow]"
        )

        # TODO: Implement LLM query parsing
        # For now, show placeholder
        console.print(
            "[yellow]⚠ LLM query parsing not yet integrated in CLI mode.[/yellow]"
        )
        console.print(
            "[dim]Use interactive mode instead: `atomic` or `atomic interactive`[/dim]\n"
        )

        console.print("\n[bold yellow]📋 What this will do:[/bold yellow]")
        console.print("  1. Parse query with LLM to extract parameters")
        console.print("  2. Show parsed parameters in a table")
        console.print("  3. Generate/retrieve structure")
        console.print("  4. Optional: Relax structure with MACE")
        console.print("  5. Run microscopy simulation")
        console.print("  6. Save results to output folder")
        console.print("\n[dim]Coming soon![/dim]\n")

    except Exception as e:
        # Print error without markup to avoid Rich parsing issues
        console.print("\n[bold red]✗ Error:[/bold red]")
        console.print(str(e), style="red", markup=False)
        console.print()
        logger.error(f"Query processing failed: {e}", exc_info=True)
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
        console.print(str(e), style="red", markup=False)
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
