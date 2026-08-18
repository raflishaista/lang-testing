import base64
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import pymupdf

load_dotenv()

LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def pdf_to_base64(pdf_path: str, dpi: int = 200) -> list[dict]:
    """Render each PDF page to a base64-encoded PNG image."""
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc):
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        images.append({
            "page": page_num + 1,
            "total": len(doc),
            "b64": b64,
        })
    doc.close()
    return images


@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF by OCR-ing each page and save to a timestamped .txt file."""
    path = Path(pdf_path)
    if not path.exists():
        return f"Error: File not found: {pdf_path}"

    model = ChatOpenAI(
        model="ocr-lighton",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )

    page_images = pdf_to_base64(str(path))
    all_text_parts = []

    for img in page_images:
        response = model.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extract all text from page {img['page']} of {img['total']}. Return ONLY the text, nothing else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['b64']}"}},
                ],
            }
        ])
        all_text_parts.append(f"--- Page {img['page']} ---\n{response.content.strip()}")

    full_text = "\n\n".join(all_text_parts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"ocr_pdf_{timestamp}_{path.stem}.txt"
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(full_text, encoding="utf-8")

    return f"Extracted {len(page_images)} pages. Saved to: {out_path}\n\n{full_text}"


@tool
def extract_text_from_pdf_folder(folder_path: str) -> str:
    """Extract text from all PDFs in a folder and save OCR outputs to ocr_output/."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: Not a directory: {folder_path}"

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        return f"No PDF files found in {folder_path}"

    results = []
    for pdf_file in pdf_files:
        result = extract_text_from_pdf.invoke(str(pdf_file))
        results.append(result)

    return "\n\n=== ===\n\n".join(results)


SYSTEM_PROMPT = """You are a helpful PDF OCR assistant. Your job is to extract text from PDF files.

When given a PDF path or folder, use the appropriate tool to extract all text from every page.
Return the result clearly to the user, including where the output file was saved."""


def pdf_agent(pdf_path: str) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[extract_text_from_pdf],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [("user", f"Extract text from this PDF: {pdf_path}")]},
        stream_mode="values",
    )
    return result["messages"][-1].content


def pdf_folder_agent(folder_path: str) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[extract_text_from_pdf, extract_text_from_pdf_folder],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [("user", f"Extract text from all PDFs in folder: {folder_path}")]},
        stream_mode="values",
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    # Set UTF-8 output for Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python langchain_pdf.py <pdf_path|folder_path>")
        print("  Folder mode: scans all .pdf files in the folder")
        print("  File mode: processes a single PDF")
        sys.exit(1)

    target = sys.argv[1]
    if Path(target).is_dir():
        print(f"[Folder mode] Scanning: {target}")
        result = pdf_folder_agent(target)
    else:
        print(f"[File mode] Processing: {target}")
        result = pdf_agent(target)

    print(result)
