import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_pdf import pdf_agent, extract_text_from_pdf
from langchain_explain import read_ocr_output

load_dotenv()

LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")
OCR_DIR = Path(__file__).parent / "ocr_output"
OCR_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "explanation_output"
OUTPUT_DIR.mkdir(exist_ok=True)


@tool
def search_web(query: str) -> str:
    """Search the web for additional context or explanation when needed."""
    return DuckDuckGoSearchRun(name="web_search").invoke(query)


@tool
def list_ocr_files(folder_path: str = None) -> str:
    """List all OCR output text files."""
    base = Path(__file__).parent / "ocr_output"
    if folder_path:
        base = Path(folder_path)
    files = sorted(base.glob("*.txt"), reverse=True)
    if not files:
        return "No OCR files found."
    return "\n".join(f"{i+1}. {f.name} ({f.stat().st_size} bytes)" for i, f in enumerate(files))


@tool
def list_pdf_files(folder_path: str) -> str:
    """List all PDF files in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: Not a directory: {folder_path}"
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        return f"No PDF files found in {folder_path}"
    return "\n".join(f"{i+1}. {p.name} ({p.stat().st_size} bytes)" for i, p in enumerate(pdfs))


SYSTEM_PROMPT = """You are a document query assistant. Your job is to read documents (PDFs, OCR output text files) and answer/explain the questions or words in them.

Workflow:
1. Use the provided document content as your primary source
2. Use `search_web` if additional context is needed to explain a term, reference, or concept
3. Generate a clear, concise answer that addresses the question

Be thorough but focused. If asked to "explain" something, provide context and significance."""


def query_agent(content: str, question: str) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[search_web],
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = f"""Document content:
---
{content}
---

Question: {question}

Use web search if you need additional context to answer the question."""

    result = agent.invoke(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    )
    return result["messages"][-1].content


def query_pdf(pdf_path: str, question: str = None) -> str:
    """Orchestrator: extract text from PDF via sub-agent, then answer the question."""
    question = question or "Summarize and explain the content of this PDF."
    print(f"[1/2] Extracting text from PDF: {pdf_path}")
    extracted_text = pdf_agent(pdf_path)
    print(f"[1/2] Text extracted ({len(extracted_text)} chars)")

    print(f"[2/2] Answering question via query agent...")
    answer = query_agent(extracted_text, question)
    _save_output(answer, f"query_{Path(pdf_path).stem}.txt")
    return answer


def query_ocr(file_path: str, question: str = None) -> str:
    """Orchestrator: read an existing OCR output text file, then answer the question."""
    question = question or "Summarize and explain the content."
    print(f"[1/2] Reading OCR output: {file_path}")
    content = read_ocr_output.invoke(file_path)
    print(f"[1/2] Content loaded ({len(content)} chars)")

    print(f"[2/2] Answering question via query agent...")
    answer = query_agent(content, question)
    _save_output(answer, f"query_{Path(file_path).stem}.txt")
    return answer


def query_folder(folder_path: str, question: str = None, mode: str = "pdf") -> str:
    """Orchestrator: process all matching files in a folder."""
    question = question or "Summarize and explain the content."
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: Not a directory: {folder_path}"

    if mode == "pdf":
        files = sorted(folder.glob("*.pdf"))
        process_fn = lambda f: query_pdf(str(f), question)
    else:
        files = sorted(folder.glob("*.txt"))
        process_fn = lambda f: query_ocr(str(f), question)

    if not files:
        return f"No {mode} files found in {folder_path}"

    results = []
    for f in files:
        print(f"\n{'='*60}")
        print(f"Processing: {f.name}")
        print(f"{'='*60}")
        result = process_fn(f)
        results.append(f"\n### {f.name} ###\n{result}")

    return "\n\n".join(results)


def _save_output(content: str, filename: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{timestamp}_{filename}"
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(content, encoding="utf-8")
    print(f"Output saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python langchain_query.py pdf <pdf_path|folder> [question]")
        print("  python langchain_query.py ocr <ocr_txt_path|folder> [question]")
        sys.exit(1)

    mode = sys.argv[1]
    target = sys.argv[2]
    question = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "pdf":
        if Path(target).is_dir():
            print(f"[Folder mode] Scanning: {target}")
            result = query_folder(target, question, mode="pdf")
        else:
            result = query_pdf(target, question)
    elif mode == "ocr":
        if Path(target).is_dir():
            print(f"[Folder mode] Scanning: {target}")
            result = query_folder(target, question, mode="ocr")
        else:
            result = query_ocr(target, question)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(result)
