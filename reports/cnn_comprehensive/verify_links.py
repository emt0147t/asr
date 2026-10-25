import os
import re

BASE_DIR = r"d:\trainidentificationaudio"
DOCS_DIR = os.path.join(BASE_DIR, "reports", "cnn_comprehensive")

link_pattern = re.compile(r'\[([^\]]+)\]\(file:///([^\)]+)\)')

total_links = 0
broken_links = 0

for root, dirs, files in os.walk(DOCS_DIR):
    for f in files:
        if f.endswith(".md"):
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8") as md_f:
                content = md_f.read()
            matches = link_pattern.findall(content)
            for text, link in matches:
                total_links += 1
                # clean link: strip anchor #L... if any
                clean_link = link.split('#')[0]
                # Windows path normalizer
                win_path = clean_link.replace('/', '\\')
                if not os.path.exists(win_path):
                    print(f"BROKEN LINK in {f}: {link} -> {win_path}")
                    broken_links += 1

print(f"Verification complete: {total_links} links checked, {broken_links} broken links.")
