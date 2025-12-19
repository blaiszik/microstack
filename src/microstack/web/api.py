from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
from pathlib import Path

from microstack.agents.workflow import run_workflow
from microstack.agents.session_manager import get_session_state
from microstack.utils.config import OUTPUT_DIR, LOG_FILE

app = FastAPI(title="µStack API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Returns the last N lines of the log file."""
    if not LOG_FILE.exists():
        return {"logs": "Log file not found."}

    try:
        with open(LOG_FILE, "r") as f:
            # Simple tail implementation
            all_lines = f.readlines()
            last_lines = all_lines[-lines:]
            return {"logs": "".join(last_lines)}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())[:8]

    try:
        # Run the workflow
        final_state = run_workflow(request.query, session_id)

        # Generate reports after workflow completion (similar to interactive.py)
        report_md = ""
        ai_summary_md = ""
        all_images = []

        structure_dir = None
        if final_state.file_paths.get("structure_dir"):
            structure_dir = Path(final_state.file_paths["structure_dir"])
        elif final_state.file_paths.get("output_dir"):
            structure_dir = Path(final_state.file_paths["output_dir"]).parent

        if structure_dir and structure_dir.exists():
            # 1. Generate full report
            from microstack.utils.report_generator import generate_full_report

            report_md = generate_full_report(final_state, structure_dir)

            # 2. Generate AI summary if relaxation results exist
            if final_state.relaxation_results and final_state.structure_info:
                from microstack.relaxation.relax_report_generator import (
                    generate_natural_description,
                )

                formula = final_state.structure_info.get("formula", "Unknown")
                analysis = {
                    "energy_change_eV": final_state.relaxation_results.get(
                        "energy_change", 0
                    ),
                    "relaxation": {
                        "n_atoms": final_state.structure_info.get("num_atoms", 0),
                    },
                    "microscopy_results": final_state.microscopy_results or {},
                }
                ai_summary_md = generate_natural_description(
                    formula, "surface", analysis
                )
                summary_file = structure_dir / f"{formula}_ai_summary.md"
                with open(summary_file, "w") as f:
                    f.write(
                        f"# {formula} Surface Relaxation Summary\n\n{ai_summary_md}"
                    )

            # 3. Scan for all images and structure files in the structure directory recursively
            for asset_path in structure_dir.rglob("*"):
                if asset_path.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".xyz",
                ]:
                    all_images.append(str(asset_path.absolute()))

        summary = final_state.get_summary()

        results = {
            "workflow_summary": summary,
            "errors": final_state.errors,
            "warnings": final_state.warnings,
            "structure_info": final_state.structure_info,
            "microscopy_results": final_state.microscopy_results,
            "file_paths": final_state.file_paths,
            "relaxation_results": final_state.relaxation_results,
            "report_md": report_md,
            "ai_summary_md": ai_summary_md,
            "all_images": all_images,
        }

        return QueryResponse(
            session_id=session_id,
            status="success",
            message="Workflow completed successfully",
            results=results,
        )
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    state = get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.get_summary()


# Mount output directory to serve images and other files
if OUTPUT_DIR.exists():
    app.mount(
        "/output", StaticFiles(directory=str(OUTPUT_DIR.absolute())), name="output"
    )


@app.get("/")
async def root():
    return {"message": "µStack API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
