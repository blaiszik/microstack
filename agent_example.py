
import sys
import operator
import warnings
import os

# Suppress warnings to keep the CLI clean
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from typing import Annotated, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import our local modules
import generate_surfaces
import surface_relaxation
import comparison
import report_generator
import materials_project
import config
from ase.io import write

# Rich for markdown rendering
try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Define State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    surfaces: dict

# --- Tools ---


@tool
def generate_surface(element: str, face: str) -> str:
    """
    Generates an atomic surface.
    
    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt', 'Au', 'C', 'MoS2')
        face: Surface face ('100', '111', '110', 'graphene', '2d')
        
    Returns:
        A message indicating success and the filename of the generated surface.
    """
    try:
        atoms = generate_surfaces.create_surface(element, face)
        filename = f"{element}_{face}_unrelaxed.xyz"
        write(filename, atoms)
        return f"Successfully generated {element}({face}) surface with {len(atoms)} atoms. Saved to {filename}."
    except Exception as e:
        return f"Error generating surface: {str(e)}"

@tool
def relax_surface(element: str, face: str) -> str:
    """
    Relaxes an atomic surface using a machine learning potential (MACE).

    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt', 'Au', 'C', 'MoS2')
        face: Surface face ('100', '111', '110', 'graphene', '2d')

    Returns:
        A message indicating success, energy changes, and the filename of the relaxed surface.
    """
    try:
        # Re-generate to ensure we have the object
        atoms = generate_surfaces.create_surface(element, face)

        print("\033[90mLoading model...\033[0m")
        # Suppress stdout/stderr from model loading if possible, but it's hard to catch C++ level logs.
        # We can at least try to be quiet.
        model = surface_relaxation.load_model()

        print(f"\033[96mRelaxing {element}({face})...\033[0m")
        relaxed_surfaces, initial_energies, final_energies = surface_relaxation.relax_surfaces(
            [atoms], model, steps=50
        )

        relaxed_atoms = relaxed_surfaces[0]
        init_e = initial_energies[0]
        final_e = final_energies[0]

        filename = f"{element}_{face}_relaxed.xyz"
        write(filename, relaxed_atoms)

        # Also generate the plot
        surface_relaxation.plot_surface_relaxation(
            [atoms], [relaxed_atoms], [f"{element}({face})"], filename=f"{element}_{face}_relaxation.png"
        )

        return (f"Relaxation complete.\n"
                f"Initial Energy: {init_e:.4f} eV\n"
                f"Final Energy: {final_e:.4f} eV\n"
                f"Change: {final_e - init_e:.4f} eV\n"
                f"Structure saved to {filename}\n"
                f"Visualization saved to {element}_{face}_relaxation.png")

    except Exception as e:
        return f"Error relaxing surface: {str(e)}"


@tool
def analyze_surface(element: str, face: str) -> str:
    """
    Full analysis pipeline: generate, relax, compare with experiments, and generate AI report.

    This is the main tool for comprehensive surface analysis. It:
    1. Generates the surface structure
    2. Relaxes it using MACE ML potential
    3. Compares with Materials Project and literature data
    4. Generates a scientific report with AI-powered discussion

    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt', 'Au', 'Ni', 'MoS2')
        face: Surface face ('100', '111', '110', 'graphene', '2d')

    Returns:
        A comprehensive markdown report with analysis and scientific discussion.
    """
    try:
        # Step 1: Generate surface
        print("\033[96m[1/5] Generating surface structure...\033[0m")
        atoms = generate_surfaces.create_surface(element, face)
        unrelaxed = atoms.copy()

        unrelaxed_file = f"{element}_{face}_unrelaxed.xyz"
        write(unrelaxed_file, atoms)
        print(f"      Created {element}({face}) with {len(atoms)} atoms")

        # Step 2: Load model and relax
        print("\033[96m[2/5] Loading MACE model...\033[0m")
        model = surface_relaxation.load_model()

        print(f"\033[96m[3/5] Relaxing surface...\033[0m")
        relaxed_surfaces, initial_energies, final_energies = surface_relaxation.relax_surfaces(
            [atoms], model, steps=config.DEFAULT_RELAXATION_STEPS
        )

        relaxed = relaxed_surfaces[0]
        init_e = initial_energies[0]
        final_e = final_energies[0]

        relaxed_file = f"{element}_{face}_relaxed.xyz"
        write(relaxed_file, relaxed)
        print(f"      Energy: {init_e:.4f} → {final_e:.4f} eV (Δ = {final_e - init_e:.4f} eV)")

        # Step 3: Generate visualization
        print("\033[96m[4/5] Generating visualization...\033[0m")
        figure_file = f"{element}_{face}_relaxation.png"
        surface_relaxation.plot_surface_relaxation(
            [unrelaxed], [relaxed], [f"{element}({face})"], filename=figure_file
        )

        # Step 4: Full analysis with comparison
        print("\033[96m[5/5] Analyzing and generating report...\033[0m")
        analysis = comparison.full_analysis(
            unrelaxed=unrelaxed,
            relaxed=relaxed,
            element=element,
            face=face,
            initial_energy=init_e,
            final_energy=final_e
        )

        # Check for reference data
        if analysis['comparison']['has_reference']:
            print(f"      Comparing with: {analysis['comparison']['reference_source']}")
            print(f"      Agreement: {analysis['comparison']['overall_agreement'].upper()}")
        else:
            print("      No reference data available for comparison")

        # Step 5: Generate report with AI discussion
        print("\033[90m      Generating AI discussion...\033[0m")
        report = report_generator.generate_full_report(
            element=element,
            face=face,
            analysis=analysis,
            figure_paths=[figure_file]
        )

        # Save report
        report_file = f"{element}_{face}_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"\033[92m✓ Analysis complete! Report saved to {report_file}\033[0m")

        return report

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during analysis: {str(e)}"


@tool
def list_references() -> str:
    """
    List all available surface reference data for comparison.

    Returns:
        A formatted list of elements and surfaces with reference data.
    """
    refs = materials_project.list_available_references()

    lines = ["# Available Reference Data\n"]
    lines.append("The following surfaces have experimental/DFT reference data for comparison:\n")

    for element, faces in sorted(refs.items()):
        lines.append(f"- **{element}**: {', '.join(faces)}")

    lines.append("\nUse `analyze <element> <face>` for full analysis with comparison.")

    return "\n".join(lines)

# --- Agent Logic ---

def parse_element_face(content: str) -> tuple[str, str]:
    """Parse element and face from natural language input."""
    known_elements = ['Cu', 'Pt', 'Au', 'Ag', 'Al', 'Ni', 'Pd', 'Fe', 'Ir', 'Rh', 'C', 'MoS2', 'WS2']
    known_faces = ['100', '111', '110', 'graphene', '2d']

    words = content.lower().split()
    element = "Cu"  # Default
    face = "100"    # Default

    for w in words:
        # Check for graphene special case
        if w == 'graphene':
            element = 'C'
            face = 'graphene'
            break

        # Check elements
        w_cap = w.capitalize()
        if w_cap in known_elements:
            element = w_cap
        elif w.upper() in ['MOS2', 'WS2']:
            if w.lower() == 'mos2':
                element = 'MoS2'
            if w.lower() == 'ws2':
                element = 'WS2'

        # Check faces
        if w in known_faces:
            face = w

    return element, face


def heuristic_agent(state: AgentState):
    """
    A simple heuristic 'model' that parses natural language to tool calls.
    """
    last_message = state['messages'][-1]
    content = last_message.content.lower()

    # Check for list references command
    if "list" in content and ("reference" in content or "available" in content):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "list_references", "args": {}, "id": "call_1"}
        ])]}

    # Check for full analysis (highest priority)
    if "analyze" in content or "analysis" in content or "report" in content or "compare" in content:
        element, face = parse_element_face(content)
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "analyze_surface", "args": {"element": element, "face": face}, "id": "call_1"}
        ])]}

    # Check for relax
    if "relax" in content:
        element, face = parse_element_face(content)
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "relax_surface", "args": {"element": element, "face": face}, "id": "call_1"}
        ])]}

    # Check for generate
    if "generate" in content or "create" in content:
        element, face = parse_element_face(content)
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "generate_surface", "args": {"element": element, "face": face}, "id": "call_1"}
        ])]}

    # Help message
    return {"messages": [AIMessage(content="""I can help you with surface analysis! Try:

• **analyze Cu 100** - Full analysis with ML relaxation, comparison to experiments, and AI-generated report
• **relax Pt 111** - Quick relaxation without full analysis
• **generate graphene** - Just create the structure
• **list references** - Show available experimental data

Supported elements: Cu, Pt, Au, Ag, Ni, Pd, MoS2
Supported faces: 100, 111, 110, graphene, 2d""")]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", heuristic_agent)
tools = [generate_surface, relax_surface, analyze_surface, list_references]
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Add edges
workflow.set_entry_point("agent")

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

def heuristic_agent_with_response(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage):
        return {"messages": [AIMessage(content=f"Done! {last_message.content}")]}
    
    return heuristic_agent(state)

# Re-build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", heuristic_agent_with_response)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- CLI Interface ---

def print_logo():
    # Colors
    C_CYAN = "\033[96m"
    C_BLUE = "\033[94m"
    C_YELLOW = "\033[93m"
    C_RESET = "\033[0m"

    logo = f"""
{C_BLUE}      .       .       *       .       .{C_RESET}
{C_BLUE}    *   .   *   .   *   .   *   .{C_RESET}     {C_CYAN}   ___   __________  __  ___________{C_RESET}
{C_BLUE}      .   .   .   .   .   .   .{C_RESET}       {C_CYAN}  / _ | /_  __/ __ \\/  |/  /  _/ ___/{C_RESET}
{C_BLUE}    .   *   .   *   .   *   .   *{C_RESET}     {C_CYAN} / __ |  / / / /_/ / /|_/ // // /__  {C_RESET}
{C_BLUE}      .   .   .   .   .   .   .{C_RESET}       {C_CYAN}/_/ |_| /_/  \\____/_/  /_/___/\\___/  {C_RESET}
{C_BLUE}    AI MATERIALS SCIENTIST {C_RESET}
    """
    print(logo)


def render_markdown(text: str):
    """Render markdown text in the terminal."""
    if RICH_AVAILABLE and console:
        md = Markdown(text)
        console.print(md)
    else:
        # Fallback: just print the text
        print(text)

def main():
    print_logo()
    print("\033[1mWelcome to the AI Materials Scientist!\033[0m")
    print("I can analyze atomic surfaces using ML potentials and compare with experiments.")
    print("")
    print("Try: '\033[93manalyze Cu 100\033[0m' for full analysis with AI-generated report")
    print("     '\033[93mlist references\033[0m' to see available experimental data")
    print("Type 'quit' or 'exit' to leave.\n")

    # Check config
    warnings = config.validate_config()
    if warnings:
        print("\033[93mConfiguration warnings:\033[0m")
        for w in warnings:
            print(f"  ⚠ {w}")
        print("")

    while True:
        try:
            user_input = input("\033[92mUser> \033[0m")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            inputs = {"messages": [HumanMessage(content=user_input)]}

            print("\033[90mProcessing...\033[0m")

            final_response = None
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "agent":
                        msg = value["messages"][0]
                        if isinstance(msg, AIMessage) and not msg.tool_calls:
                            final_response = msg.content
                    elif key == "tools":
                        # Tool output - this is where the report comes from
                        for tool_msg in value.get("messages", []):
                            if isinstance(tool_msg, ToolMessage):
                                final_response = tool_msg.content

            if final_response:
                print("")
                # Check if it looks like markdown (has headers or tables)
                if final_response.startswith("#") or "|" in final_response or "**" in final_response:
                    render_markdown(final_response)
                else:
                    print(f"\033[94m{final_response}\033[0m")
                print("")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    main()