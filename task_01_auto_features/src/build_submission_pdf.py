"""Render the Russian submission Markdown to an A4 PDF (publication only)."""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                              TableStyle, Image, KeepTogether)

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
SOURCE = TASK / "report/SUBMISSION_REPORT_RU.md"
OUTPUT = REPO / "output/pdf/T-Lab_Report_RU.pdf"


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#176276">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font color="#45566b">\1</font>', text)
    return text


def main():
    fonts = Path("C:/Windows/Fonts")
    for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                       ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                italic="Arial-Italic", boldItalic="Arial-BoldItalic")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyRU", fontName="Arial", fontSize=9.5,
                             leading=13.7, spaceAfter=7, allowWidows=0, allowOrphans=0,
                             textColor=colors.HexColor("#253247")))
    styles.add(ParagraphStyle(name="TitleRU", fontName="Arial-Bold", fontSize=23,
                             leading=28, spaceAfter=15, textColor=colors.HexColor("#16354a")))
    styles.add(ParagraphStyle(name="H2RU", fontName="Arial-Bold", fontSize=13,
                             leading=17, spaceBefore=14, spaceAfter=8,
                             keepWithNext=True, textColor=colors.HexColor("#126173")))
    styles.add(ParagraphStyle(name="H3RU", fontName="Arial-Bold", fontSize=10.5,
                             leading=14, spaceBefore=9, spaceAfter=6,
                             keepWithNext=True, textColor=colors.HexColor("#253247")))
    styles.add(ParagraphStyle(name="CellRU", fontName="Arial", fontSize=7.7,
                             leading=10.1, spaceAfter=0))
    styles.add(ParagraphStyle(name="CodeRU", fontName="Arial", fontSize=8.3,
                             leading=12, backColor=colors.HexColor("#eef3f6"),
                             borderPadding=8, spaceAfter=8))
    width = 174 * mm
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["TitleRU"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H2RU"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["H3RU"]))
        elif line.startswith("!["):
            path = TASK / "report/submission_comparison_ru.png"
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                w, h = im.size
            figure = [Image(str(path), width=width, height=width * h / w),
                      Spacer(1, 3 * mm)]
            caption_index = i + 1
            while caption_index < len(lines) and not lines[caption_index].strip():
                caption_index += 1
            if caption_index < len(lines) and lines[caption_index].startswith("Рисунок "):
                figure.append(Paragraph(inline(lines[caption_index]), styles["BodyRU"]))
                i = caption_index
            story.append(KeepTogether(figure))
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(html.escape(lines[i]))
                i += 1
            story.append(Paragraph("<br/>".join(code), styles["CodeRU"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            columns = len(rows[0])
            weights = [1.6] + [1] * (columns - 1)
            if columns == 3:
                weights = [1.15, 2.5, 1.65]
            if columns == 2:
                weights = [1, 3.2]
            if columns == 3 and rows[0][0] == "Группа":
                weights = [1.2, 2.8, 0.8]
            data = [[Paragraph(inline(c), styles["CellRU"]) for c in row] for row in rows]
            table = Table(data, colWidths=[width * w / sum(weights) for w in weights],
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0edf2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#47839a")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([KeepTogether([table]), Spacer(1, 3 * mm)])
            continue
        else:
            text = line
            if text.startswith("- "):
                text = "• " + text[2:]
            story.append(Paragraph(inline(text), styles["BodyRU"]))
        i += 1

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(colors.HexColor("#607381"))
        canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(210 * mm, 297 * mm),
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=23 * mm,
                            title="Автоматическая генерация и отбор признаков для музыкальных рекомендаций",
                            author="T-Lab", subject="Исследовательская работа 1, P001-P008")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
