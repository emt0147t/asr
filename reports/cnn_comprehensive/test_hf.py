import os
import subprocess
import pypdf

test_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {
    size: A4 portrait;
    margin: 20mm 15mm 20mm 15mm;
}
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
}
.page-break {
    page-break-after: always;
}
</style>
</head>
<body>
<h1>Trang 1: Bìa Báo Cáo KWS</h1>
<p>Kiểm tra header footer template trong Edge.</p>
<div class="page-break"></div>
<h1>Trang 2: Nội Dung Chính</h1>
<p>Nội dung trang 2.</p>
</body>
</html>
"""

with open("test_hf.html", "w", encoding="utf-8") as f:
    f.write(test_html)

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_full = os.path.abspath("test_hf.html")
pdf_full = os.path.abspath("test_hf.pdf")
html_url = "file:///" + html_full.replace("\\", "/")

header_tpl = '<div style="font-size:8pt;width:100%;text-align:right;color:#888;padding-right:15mm;font-family:\'Segoe UI\',sans-serif;">Vietnamese KWS CNN &mdash; Báo Cáo Tổng Quan</div>'
footer_tpl = '<div style="font-size:8pt;width:100%;text-align:center;color:#666;font-family:\'Segoe UI\',sans-serif;">Trang <span class="pageNumber"></span> / <span class="totalPages"></span></div>'

cmd = [
    edge_path,
    "--headless=new",
    "--disable-gpu",
    "--display-header-footer",
    f"--header-template={header_tpl}",
    f"--footer-template={footer_tpl}",
    "--generate-pdf-document-outline",
    f"--print-to-pdf={pdf_full}",
    html_url
]

subprocess.run(cmd, check=True)

reader = pypdf.PdfReader(pdf_full)
print(f"Num pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"Page {i+1} lines count: {len(text.splitlines())}")

if os.path.exists("test_hf.html"):
    os.remove("test_hf.html")
if os.path.exists("test_hf.pdf"):
    os.remove("test_hf.pdf")
