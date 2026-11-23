"""Build an editable client discussion deck from Bob's six Markdown slides."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

from presentation import SLIDE


ROOT = Path(__file__).resolve().parent
RED = RGBColor(237, 28, 46)  # Official SteriMax wordmark red.
BLACK = RGBColor(35, 31, 32)
MUTED = RGBColor(93, 92, 94)
PALE = RGBColor(248, 247, 246)
WHITE = RGBColor(255, 255, 255)


def plain(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("*", "").strip()


def box(slide, x, y, w, h, fill=WHITE, radius=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def label(slide, text, x, y, w, h, size=22, color=BLACK, bold=False, margin=0, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.text = plain(text)
    p.font.name = "Arial"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.space_after = Pt(0)
    return shape


def split_deck(markdown: str):
    headings = list(SLIDE.finditer(markdown))
    if len(headings) != 6:
        raise ValueError("PowerPoint requires exactly six Bob slides")
    slides = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = markdown[heading.end():end]
        points = [line.strip()[2:] for line in body.splitlines() if line.strip().startswith("- ")]
        if not points:
            raise ValueError(f"slide {i + 1} has no content")
        slides.append((heading.group(2).strip(), points, body))
    return slides


def render_deck_pptx(markdown: str, solution_direction: str, mission: str, output: Path) -> None:
    slides = split_deck(markdown)
    branded = "sterimax" in mission.lower()
    logo = ROOT / "assets" / "sterimax-logo.png" if branded else None
    if branded and not logo.is_file():
        raise FileNotFoundError("official SteriMax logo asset is missing")
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)

    for index, (title, points, raw) in enumerate(slides):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        box(slide, 0, 0, 13.333, 7.5)
        box(slide, 0, 0, 0.16, 7.5, RED)
        if logo:
            slide.shapes.add_picture(str(logo), Inches(0.52), Inches(0.3), width=Inches(2.06))
        else:
            label(slide, "CLIENT DISCUSSION", 0.55, 0.34, 3.8, 0.35, 13, RED, True)
        label(slide, "REVIEW DRAFT · FOR HUMAN APPROVAL", 9.75, 0.36, 3.05, 0.26, 10, MUTED)
        label(slide, title, 0.62, 1.03, 12.0, 0.9, 34, BLACK, True)
        box(slide, 0.62, 1.93, 0.78, 0.065, RED)

        if index == 0:
            positions = [(0.62, 2.38), (6.71, 2.38), (0.62, 4.47), (6.71, 4.47)]
            for j, point in enumerate(points[:4]):
                x, y = positions[j]
                box(slide, x, y, 5.83, 1.78, PALE, True)
                box(slide, x, y, 0.07, 1.78, RED)
                label(slide, point, x + 0.23, y + 0.19, 5.38, 1.43,
                      18 if len(plain(point)) < 170 else 16)
        elif index == 1:
            for j, point in enumerate(points[:2]):
                x = 0.62 + j * 6.11
                box(slide, x, 2.32, 5.87, 3.48, PALE, True)
                label(slide, f"0{j + 1} / HYPOTHESIS", x + 0.25, 2.58, 5.3, 0.35, 14, RED, True)
                label(slide, point, x + 0.25, 3.12, 5.28, 2.43,
                      20 if len(plain(point)) < 290 else 18)
            if len(points) > 2:
                label(slide, points[2], 0.69, 6.08, 11.9, 0.55, 17, MUTED)
        elif index == 2:
            for j, point in enumerate(points[:3]):
                y = 2.26 + j * 1.42
                box(slide, 0.62, y, 1.12, 1.08, RED if j == 0 else BLACK, True)
                label(slide, f"0{j + 1}", 0.93, y + 0.29, 0.7, 0.5, 24, WHITE, True)
                label(slide, point, 2.0, y + 0.10, 10.6, 0.95,
                      22 if len(plain(point)) < 130 else 20)
        elif index == 3:
            for j, point in enumerate(points[:3]):
                x = 0.62 + j * 4.08
                box(slide, x, 2.28, 3.79, 2.83, PALE if j != 0 else BLACK, True)
                label(slide, point, x + 0.2, 2.56, 3.39, 2.28,
                      18 if len(plain(point)) < 190 else 16,
                      WHITE if j == 0 else BLACK)
            # Sol's conditional walkthrough is drawn as editable PowerPoint shapes.
            match = re.search(r":\s*([A-Za-z][A-Za-z ,]+)\.", points[0])
            stages = [part.strip().capitalize() for part in match.group(1).split(",") if part.strip()] if match else []
            stages = stages[:5]
            if len(stages) >= 3 and "walkthrough" in solution_direction.lower():
                label(slide, "IF VALIDATED · ONE REAL CASE", 0.65, 5.37, 6, 0.3, 12, RED, True)
                step_w = 2.24
                for j, stage in enumerate(stages):
                    x = 0.65 + j * 2.43
                    box(slide, x, 5.79, step_w, 0.72, WHITE, True).line.color.rgb = RED
                    label(slide, stage, x + 0.10, 5.99, step_w - 0.2, 0.34, 15, BLACK, True)
                    if j + 1 < len(stages):
                        label(slide, "→", x + 2.24, 5.99, 0.2, 0.35, 16, RED, True)
        elif index == 4:
            for j, point in enumerate(points[:3]):
                x = 0.62 + j * 4.09
                label(slide, f"0{j + 1}", x, 2.32, 1.3, 0.78, 45, RED, True)
                box(slide, x, 3.14, 3.65, 0.045, BLACK)
                label(slide, point, x, 3.46, 3.55, 2.85,
                      18 if len(plain(point)) < 130 else 17)
        else:
            box(slide, 0.62, 2.28, 11.98, 2.09, BLACK, True)
            label(slide, points[0], 0.98, 2.65, 11.15, 1.4, 27, WHITE, True)
            for j, point in enumerate(points[1:3]):
                x = 0.67 + j * 6.0
                box(slide, x, 4.72, 5.75, 1.72, PALE, True)
                label(slide, point, x + 0.22, 4.94, 5.29, 1.32,
                      18 if len(plain(point)) < 135 else 17)

        urls = re.findall(r"https?://[^\s)]+", raw)
        domains = ", ".join(dict.fromkeys(urlsplit(u).netloc.removeprefix("www.") for u in urls))
        footer = "Source: " + domains if domains else "Discussion material · verify hypotheses with the client"
        label(slide, footer, 0.62, 7.05, 10.5, 0.23, 10, MUTED)
        label(slide, f"{index + 1:02d} / 06", 11.82, 7.05, 0.8, 0.23, 10, MUTED)
        slide.notes_slide.notes_text_frame.text = "Bob's source text:\n" + raw.strip() + "\n\nSol's direction:\n" + solution_direction

    output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(output)
