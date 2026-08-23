import subprocess
import os
import sys
from pypdf import PdfReader, PdfWriter
import xml.etree.ElementTree as ET


def get_browser():
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in edge_paths:
        if os.path.exists(p):
            return p
    raise RuntimeError("No browser found for SVG to PDF conversion")


def convert_svg_file(svg_path, pdf_path):
    browser = get_browser()
    svg_path = os.path.abspath(svg_path)
    pdf_path = os.path.abspath(pdf_path)

    tree = ET.parse(svg_path)
    root = tree.getroot()
    viewbox = root.get("viewBox")
    if viewbox:
        parts = viewbox.split()
        vw, vh = float(parts[2]), float(parts[3])
    else:
        width_str = root.get("width", "800px").replace("px", "").replace("pt", "")
        height_str = root.get("height", "600px").replace("px", "").replace("pt", "")
        vw, vh = float(width_str), float(height_str)

    print(
        f"Converting {os.path.basename(svg_path)} -> {os.path.basename(pdf_path)} ({vw}x{vh})"
    )

    html_file = os.path.join(
        os.path.dirname(pdf_path), f"temp_{os.path.basename(pdf_path)}.html"
    )
    svg_uri = svg_path.replace("\\", "/")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
<style>
@page {{
    size: {vw}px {vh}px;
    margin: 0;
}}
html, body {{
    margin: 0;
    padding: 0;
    width: {vw}px;
    height: {vh}px;
    overflow: hidden;
    background: transparent;
}}
img {{
    width: {vw}px;
    height: {vh}px;
    display: block;
}}
</style>
</head>
<body>
<img src="file:///{svg_uri}" />
</body>
</html>""")

    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_file,
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(html_file):
        os.remove(html_file)

    print(f"Generated {pdf_path}, size: {os.path.getsize(pdf_path)} bytes")
    reader = PdfReader(pdf_path)
    print("Page MediaBox:", reader.pages[0].mediabox)


def convert_all():
    figures_dir = os.path.abspath("icip/figures")
    for fname in os.listdir(figures_dir):
        if fname.endswith(".drawio.svg") or fname.endswith(".svg"):
            base_name = fname.replace(".drawio.svg", "").replace(".svg", "")
            svg_path = os.path.join(figures_dir, fname)
            pdf_path = os.path.join(figures_dir, f"{base_name}.pdf")
            convert_svg_file(svg_path, pdf_path)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        convert_svg_file(sys.argv[1], sys.argv[2])
    else:
        convert_all()
