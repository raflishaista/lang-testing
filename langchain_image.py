import base64
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")
OUTPUT_DIR = Path(__file__).parent / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/png")


def encode_image(image_path: str) -> tuple[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = get_mime_type(image_path)
    return mime, b64


@tool
def extract_text_from_image(image_path: str) -> str:
    """Extract all text from an image file and save it to a timestamped .txt file."""
    path = Path(image_path)
    if not path.exists():
        return f"Error: File not found: {image_path}"

    model = ChatOpenAI(
        model="ocr-lighton",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )

    mime, b64 = encode_image(str(path))
    response = model.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image exactly as it appears."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ])
    text = response.content

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"ocr_{timestamp}_{path.stem}.txt"
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(text, encoding="utf-8")

    return f"Extracted text saved to: {out_path}\n\n{text}"


@tool
def extract_text_from_image_folder(folder_path: str) -> str:
    """Extract text from all images in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: Not a directory: {folder_path}"

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])
    if not image_files:
        return f"No image files found in {folder_path}"

    results = []
    for img_file in image_files:
        result = extract_text_from_image.invoke(str(img_file))
        results.append(result)

    return "\n\n=== ===\n\n".join(results)


SYSTEM_PROMPT = """You are a helpful OCR assistant. Your job is to extract text from images.

When given an image path or folder, use the appropriate tool to extract all text.
Return the result clearly to the user, including where the output file was saved."""


def ocr_agent(image_path: str) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[extract_text_from_image],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [("user", f"Extract text from this image: {image_path}")]},
        stream_mode="values",
    )
    return result["messages"][-1].content


def ocr_folder_agent(folder_path: str) -> str:
    model = ChatOpenAI(
        model="nemotron-35",
        base_url=LLM_URL,
        api_key=LLM_KEY,
    )
    agent = create_agent(
        model=model,
        tools=[extract_text_from_image, extract_text_from_image_folder],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages": [("user", f"Extract text from all images in folder: {folder_path}")]},
        stream_mode="values",
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python langchain_image.py <image_path|folder_path>")
        print("  Folder mode: scans all .png/.jpg/.jpeg/.webp files")
        print("  File mode: processes a single image")
        sys.exit(1)

    target = sys.argv[1]
    if Path(target).is_dir():
        print(f"[Folder mode] Scanning: {target}")
        result = ocr_folder_agent(target)
    else:
        print(f"[File mode] Processing: {target}")
        result = ocr_agent(target)

    print(result)
