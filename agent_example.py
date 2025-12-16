
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
from ase.io import write

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

# --- Agent Logic ---

def heuristic_agent(state: AgentState):
    """
    A simple heuristic 'model' that parses natural language to tool calls.
    """
    last_message = state['messages'][-1]
    content = last_message.content.lower()
    
    # Common elements and faces
    known_elements = ['Cu', 'Pt', 'Au', 'Ag', 'Al', 'Ni', 'Pd', 'Fe', 'Ir', 'Rh', 'C', 'MoS2', 'WS2']
    known_faces = ['100', '111', '110', 'graphene', '2d']
    
    if "relax" in content:
        words = content.split()
        element = "Cu" # Default
        face = "100"   # Default
        
        # Heuristic parsing
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
            elif w.upper() in ['MOS2', 'WS2']: # Handle compound names
                element = w.upper() # Actually MoS2 is usually written mixed case, let's fix below
                if w.lower() == 'mos2': element = 'MoS2'
                if w.lower() == 'ws2': element = 'WS2'
            
            # Check faces
            if w in known_faces:
                face = w
        
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "relax_surface", "args": {"element": element, "face": face}, "id": "call_1"}
        ])]}
        
    elif "generate" in content or "create" in content:
        words = content.split()
        element = "Cu"
        face = "100"
        
        for w in words:
            if w == 'graphene':
                element = 'C'
                face = 'graphene'
                break
            
            w_cap = w.capitalize()
            if w_cap in known_elements:
                element = w_cap
            elif w.upper() in ['MOS2', 'WS2']:
                if w.lower() == 'mos2': element = 'MoS2'
                if w.lower() == 'ws2': element = 'WS2'
            
            if w in known_faces:
                face = w
        
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "generate_surface", "args": {"element": element, "face": face}, "id": "call_1"}
        ])]}
    
    else:
        return {"messages": [AIMessage(content="I can help you generate or relax surfaces. Try 'generate graphene' or 'relax MoS2 2d'.")]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", heuristic_agent)
tools = [generate_surface, relax_surface]
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
{C_BLUE}    SURFACE AGENT {C_RESET}
    """
    print(logo)

def main():
    print_logo()
    print("\033[1mWelcome to the Surface Agent CLI!\033[0m")
    print("I can generate and relax atomic surfaces for you.")
    print("Try: '\033[93mgenerate graphene\033[0m' or '\033[93mrelax MoS2 2d\033[0m'")
    print("Type 'quit' or 'exit' to leave.\n")
    
    messages = []
    
    while True:
        try:
            user_input = input("\033[92mUser> \033[0m")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            print("\033[90mProcessing...\033[0m")
            
            final_response = None
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "agent":
                        msg = value["messages"][0]
                        if isinstance(msg, AIMessage) and not msg.tool_calls:
                            final_response = msg.content
            
            if final_response:
                print(f"\033[94mAgent> {final_response}\033[0m")
            else:
                print("\033[94mAgent> Task completed.\033[0m")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    main()