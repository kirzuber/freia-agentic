"""Render the generated Markdown deck as a standalone HTML presentation."""

import base64
import html
import re
import xml.etree.ElementTree as ET


SLIDE = re.compile(r"^## Slide (\d+)\s*[—–:-]\s*(.+)$", re.MULTILINE)


def render_deck_html(markdown: str, diagram_svg: str) -> str:
    root = ET.fromstring(diagram_svg)
    allowed = {"svg", "g", "rect", "line", "path", "circle", "ellipse", "polyline", "polygon", "text", "tspan"}
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("diagram is not SVG")
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] not in allowed:
            raise ValueError("diagram has unsupported SVG elements")
        if any(attribute.rsplit("}", 1)[-1].lower().startswith("on") or attribute.rsplit("}", 1)[-1].lower() in {"href", "style"} for attribute in element.attrib):
            raise ValueError("diagram has active or external SVG attributes")

    headings = list(SLIDE.finditer(markdown))
    if not 5 <= len(headings) <= 7:
        raise ValueError("deck must contain five to seven slides")
    diagram_src = "data:image/svg+xml;base64," + base64.b64encode(diagram_svg.encode()).decode()
    diagram_index = next((index for index, match in enumerate(headings) if re.search(r"solution|direction|approach|option", match.group(2), re.I)), len(headings) // 2)

    def inline(text: str) -> str:
        safe = html.escape(text)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        return re.sub(r"\*(.+?)\*", r"<em>\1</em>", safe)

    def body_html(text: str) -> str:
        parts = []
        in_list = False
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append("<li>" + inline(line[2:]) + "</li>")
            else:
                if in_list:
                    parts.append("</ul>")
                    in_list = False
                if line:
                    parts.append("<p>" + inline(line) + "</p>")
        if in_list:
            parts.append("</ul>")
        return "\n".join(parts)

    page = ["""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>FreIA — Discussion deck</title><style>
    :root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#090d16;color:#e8edf7}*{box-sizing:border-box}html{scroll-snap-type:y mandatory;scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 50% 0,#202c49,#090d16 55%)}.slide{min-height:100vh;scroll-snap-align:start;display:grid;place-items:center;padding:4vw}.frame{width:min(1100px,100%);min-height:min(620px,88vh);border:1px solid #33405a;border-radius:22px;background:#111a2b;padding:clamp(28px,5vw,62px);display:flex;flex-direction:column;box-shadow:0 25px 80px #0005}.meta{color:#8b9ab3;letter-spacing:.15em;text-transform:uppercase;font-size:12px}h1{font-size:clamp(29px,4vw,49px);line-height:1.1;letter-spacing:-.04em;margin:18px 0 28px}ul{margin:0;padding-left:1.25em;display:grid;gap:17px}li,p{font-size:clamp(16px,1.7vw,22px);line-height:1.45;color:#d5deec}li::marker{color:#d7f770}strong{color:#fff}.diagram{display:block;width:min(100%,760px);max-height:230px;object-fit:contain;margin:26px auto 0;border:1px solid #36475e;border-radius:12px;background:#12161c}.nav{margin-top:auto;padding-top:28px;display:flex;justify-content:space-between;align-items:center;color:#8899b4;font-size:13px}.nav a{color:#d7f770;text-decoration:none;border:1px solid #526047;border-radius:9px;padding:9px 14px}.nav a:hover{background:#27332a}@media(max-width:640px){.slide{padding:14px}.frame{min-height:calc(100vh - 28px)}ul{gap:12px}.diagram{max-height:180px}}
    </style></head><body><main>"""]
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        title = html.escape(match.group(2).strip())
        previous = f'<a href="#slide-{index}">← Previous</a>' if index else "<span></span>"
        following = f'<a href="#slide-{index + 2}">Next →</a>' if index + 1 < len(headings) else "<span></span>"
        diagram = f'<img class="diagram" src="{diagram_src}" alt="Sol’s generated solution direction diagram">' if index == diagram_index else ""
        page.append(f'<section class="slide" id="slide-{index + 1}"><div class="frame"><div class="meta">FreIA / Executive discussion · {index + 1} of {len(headings)}</div><h1>{title}</h1>{body_html(markdown[start:end])}{diagram}<div class="nav">{previous}<span>{index + 1} / {len(headings)}</span>{following}</div></div></section>')
    page.append("</main></body></html>")
    return "\n".join(page)
