import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

load_dotenv()

LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)
EXPLANATION_DIR = Path(__file__).parent / "explanation_output"
EXPLANATION_DIR.mkdir(exist_ok=True)


@tool
def read_ocr_output(file_path: str) -> str:
    """Read the content of an OCR output text file from the ocr_output directory."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    return path.read_text(encoding="utf-8")


@tool
def list_ocr_outputs(directory: str = None) -> str:
    """List all OCR output text files available."""
    base = Path(__file__).parent / "ocr_output"
    if directory:
        base = Path(directory)
    files = sorted(base.glob("*.txt"), reverse=True)
    if not files:
        return "No OCR output files found."
    return "\n".join(f"{i+1}. {f.name} ({f.stat().st_size} bytes)" for i, f in enumerate(files))


@tool
def list_ocr_outputs_by_type() -> str:
    """List OCR outputs grouped by type (image vs PDF)."""
    image_files = sorted((Path(__file__).parent / "ocr_output").glob("ocr_*.txt"))
    pdf_files = sorted((Path(__file__).parent / "ocr_output").glob("ocr_pdf_*.txt"))
    lines = ["=== Image OCR Outputs ===", *list_ocr_outputs.invoke().split("\n")[:0], ""]
    for f in reversed(image_files):
        lines.append(f"  {f.name} ({f.stat().st_size} bytes)")
    lines.append("")
    lines.append("=== PDF OCR Outputs ===")
    for f in reversed(pdf_files):
        lines.append(f"  {f.name} ({f.stat().st_size} bytes)")
    return "\n".join(lines)


SEARCH_TOOL = DuckDuckGoSearchRun(name="web_search")

SYSTEM_PROMPT = """You are a helpful document explanation assistant. Your job is to read OCR output text files and provide clear explanations and summaries.

When given a request:
1. If no specific file is mentioned, use `list_ocr_outputs` to find available files
2. Use `read_ocr_output` to read the content
3. Provide a clear explanation of what the document is about, its key points, and significance
4. Use `web_search` when additional context would help — e.g., historical background, identifying referenced entities, or clarifying technical terms

Be thorough but concise. Focus on the main ideas and context."""


def explain_agent(file_path: str = None) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[read_ocr_output, list_ocr_outputs, SEARCH_TOOL],
        system_prompt=SYSTEM_PROMPT,
    )

    if file_path:
        prompt = f"Please explain and summarize this OCR output file: {file_path}"
    else:
        prompt = "List available OCR outputs and explain the most recent one."

    result = agent.invoke(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    )
    explanation = result["messages"][-1].content

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"explanation_{timestamp}.txt"
    out_path = EXPLANATION_DIR / out_name
    out_path.write_text(explanation, encoding="utf-8")
    return f"Saved to: {out_path}\n\n{explanation}"


def explain_folder(folder_path: str) -> str:
    """Explain all OCR outputs in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: Not a directory: {folder_path}"

    txt_files = sorted(folder.glob("*.txt"), reverse=True)
    if not txt_files:
        return f"No text files found in {folder_path}"

    results = []
    for txt_file in txt_files:
        print(f"Explaining: {txt_file.name}")
        explanation = explain_agent(str(txt_file))
        results.append(explanation)

    return "\n\n=== ===\n\n".join(results)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python langchain_explain.py <ocr_file|folder>")
        print("  Folder mode: explains all .txt files in the folder")
        print("  File mode: explains a single OCR output")
        sys.exit(1)

    target = sys.argv[1]
    if Path(target).is_dir():
        print(f"[Folder mode] Scanning: {target}")
        result = explain_folder(target)
    else:
        print(f"[File mode] Processing: {target}")
        result = explain_agent(target)

    print(result)
