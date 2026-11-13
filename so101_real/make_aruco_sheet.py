"""A4 sheets of ArUco tags (DICT_4X4_50) to print. Page 1: gripper tag, cube-top tags, table tags. Page 2: a size
range (10, 15, 20, 30, 40, 50 mm), to try which size the cameras still read well.

    .venv/bin/python so101_real/make_aruco_sheet.py            # -> so101_real/aruco_tags_A4.pdf

Print at 100 % / "Actual size" (NOT "fit to page"), then measure the 100 mm scale bar with a ruler. If it is not
100 mm, the printer scaled the page; the real tag sizes are then off by the same factor.

  * ID 0, gripper: four sizes. Use the biggest one that lies flat on the gripper's top and faces the top camera.
  * IDs 1-4, cube tops: a 24 mm tag on a 30 mm square (the cube's top face). Cut on the line and glue it on.
    There are two of each (spares).
  * IDs 10-13, table: 35 mm, for the mat corners (optional, for checking the camera later).
  * page 2, IDs 20-43: every size its own IDs (10 mm: 20-24, 15 mm: 25-28, 20 mm: 29-33, 30 mm: 34-37,
    40 mm: 38-40, 50 mm: 41-43), so a camera test can tell the sizes apart.

A tag's size is its black square, edge to edge (the white margin around it is not counted).
"""
import os
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

DPI = 600
A4_MM = (210.0, 297.0)
DICT = cv2.aruco.DICT_4X4_50
DICT_NAME = "DICT_4X4_50"
MARGIN = 12.0                    # page margin, mm

GRIPPER = [(0, s) for s in (20, 25, 30, 35)]            # (id, tag mm)
CUBE_IDS, CUBE_TAG, CUBE_FACE = (1, 2, 3, 4), 24, 30.0
TABLE_IDS, TABLE_TAG = (10, 11, 12, 13), 35
# page 2 rows: [(tag mm, how many, margin mm per side, gap mm), ...] - IDs count up from 20
SIZE_ROWS = [[(10, 5, 2.0, 4.0), (15, 4, 2.5, 5.0)], [(20, 5, 3.0, 6.0)], [(30, 4, 4.0, 8.0)],
             [(40, 3, 4.0, 8.0)], [(50, 3, 5.0, 3.0)]]
SIZE_FIRST_ID = 20


def printed_sizes():
    """{tag id: printed size mm} of every tag on the sheet (gripper ID 0 has several sizes: left out)."""
    sizes = {i: float(CUBE_TAG) for i in CUBE_IDS}
    sizes.update({i: float(TABLE_TAG) for i in TABLE_IDS})
    tag_id = SIZE_FIRST_ID
    for row in SIZE_ROWS:
        for size, count, _margin, _gap in row:
            for _ in range(count):
                sizes[tag_id] = float(size)
                tag_id += 1
    return sizes


def px(mm):
    return int(round(mm * DPI / 25.4))


def font(mm):
    return ImageFont.load_default(size=px(mm))


def marker(dictionary, tag_id, size_mm):
    """The tag's black square at exactly size_mm (6 x 6 cells: 4 x 4 bits + a 1-cell black border)."""
    cells = cv2.aruco.generateImageMarker(dictionary, tag_id, 6, borderBits=1)
    return Image.fromarray(cv2.resize(cells, (px(size_mm),) * 2, interpolation=cv2.INTER_NEAREST))


def dashed_rect(draw, x0, y0, x1, y1, dash=2.0, gap=1.5):
    """Cut line (mm coordinates)."""
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        n = max(int(np.hypot(bx - ax, by - ay) / (dash + gap)), 1)
        for i in range(n + 1):
            t0, t1 = i * (dash + gap), min(i * (dash + gap) + dash, np.hypot(bx - ax, by - ay))
            L = np.hypot(bx - ax, by - ay)
            if t0 >= L:
                break
            draw.line([(px(ax + (bx - ax) * t0 / L), px(ay + (by - ay) * t0 / L)),
                       (px(ax + (bx - ax) * t1 / L), px(ay + (by - ay) * t1 / L))], fill=150, width=px(0.2))


def tag_block(page, draw, dictionary, tag_id, size_mm, x, y, cut_mm, label=True, label_mm=2.6):
    """Tag centred in a cut_mm square whose top-left is (x, y) mm, label under it. Returns the block height."""
    off = (cut_mm - size_mm) / 2
    page.paste(marker(dictionary, tag_id, size_mm), (px(x + off), px(y + off)))
    dashed_rect(draw, x, y, x + cut_mm, y + cut_mm)
    if label:
        draw.text((px(x + cut_mm / 2), px(y + cut_mm + 1.0)), f"ID {tag_id} · {size_mm:g} mm", fill=0,
                  font=font(label_mm), anchor="mt")
    return cut_mm + 5.0


def section(draw, y, title, note):
    draw.text((px(MARGIN), px(y)), title, fill=0, font=font(4.0))
    draw.text((px(MARGIN), px(y + 5.5)), note, fill=60, font=font(2.8))
    return y + 11.0


def scale_bar(draw, x, y):
    draw.line([(px(x), px(y)), (px(x + 100), px(y))], fill=0, width=px(0.4))
    for mm in range(101):
        h = 4.0 if mm % 10 == 0 else (2.5 if mm % 5 == 0 else 1.5)
        draw.line([(px(x + mm), px(y)), (px(x + mm), px(y - h))], fill=0, width=px(0.15))
        if mm % 10 == 0:
            draw.text((px(x + mm), px(y - 5.0)), str(mm), fill=0, font=font(2.4), anchor="mb")
    draw.text((px(x + 104), px(y - 2.0)), "<- measure: must be exactly 100 mm", fill=0, font=font(3.0), anchor="lm")


def build_sizes(dictionary):
    """Page 2: the same kind of tag in 10 .. 50 mm."""
    page = Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)
    draw = ImageDraw.Draw(page)
    draw.text((px(MARGIN), px(MARGIN)), f"ArUco size test  ({DICT_NAME})", fill=0, font=font(6.0))
    draw.text((px(MARGIN), px(MARGIN + 8)), "10, 15, 20, 30, 40, 50 mm - every size has its own IDs. "
              "Print at 100 % / Actual size.", fill=0, font=font(3.0))
    scale_bar(draw, MARGIN, MARGIN + 22)
    y, tag_id = MARGIN + 30.0, SIZE_FIRST_ID
    for row in SIZE_ROWS:
        x, row_h = MARGIN, 0.0
        for k, (size, count, margin, gap) in enumerate(row):
            cut = size + 2 * margin
            if k:
                x += 5.0 - gap                       # between two sizes in one row
            for _ in range(count):
                tag_block(page, draw, dictionary, tag_id, size, x, y, cut, label_mm=2.2 if size < 15 else 2.6)
                x += cut + gap
                tag_id += 1
            row_h = max(row_h, cut)
        y += row_h + 10.0
    assert tag_id <= 50 and x <= A4_MM[0] - MARGIN + 3.0 and y <= A4_MM[1] - 10, "page 2 does not fit"
    draw.text((px(A4_MM[0] / 2), px(A4_MM[1] - 8)), "made by so101_real/make_aruco_sheet.py  -  page 2", fill=120,
              font=font(2.4), anchor="mm")
    return page


def build():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    return [build_main(dictionary), build_sizes(dictionary)]


def build_main(dictionary):
    page = Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)
    draw = ImageDraw.Draw(page)

    draw.text((px(MARGIN), px(MARGIN)), f"SO-101 ArUco tags  ({DICT_NAME})", fill=0, font=font(6.0))
    draw.text((px(MARGIN), px(MARGIN + 8)), "Print at 100 % / Actual size - NOT \"fit to page\". "
              "Tag size = the black square, edge to edge.", fill=0, font=font(3.0))
    scale_bar(draw, MARGIN, MARGIN + 22)

    # gripper: one row, four sizes, 4 mm white margin each
    y = section(draw, MARGIN + 30, "1. Gripper - ID 0",
                "Pick the biggest size that lies flat on top of the gripper, facing the top camera. Keep the white margin.")
    x, gap = MARGIN, 9.0
    h = 0.0
    for tag_id, s in GRIPPER:
        cut = s + 8.0
        h = max(h, tag_block(page, draw, dictionary, tag_id, s, x, y + (43.0 - cut), cut))
        x += cut + gap
    y += 43.0 + 7.0

    # cube tops: 2 rows of 4 (two of each id), 30 mm squares = the cube's top face
    y = section(draw, y, "2. Cube tops - IDs 1-4",
                f"{CUBE_TAG} mm tag on a {CUBE_FACE:g} mm square = the cube's top face. Cut on the dashed line, glue flat. "
                "Two of each (spares).")
    gap = 14.0
    for row in range(2):
        x = MARGIN
        for tag_id in CUBE_IDS:
            tag_block(page, draw, dictionary, tag_id, CUBE_TAG, x, y, CUBE_FACE)
            x += CUBE_FACE + gap
        y += CUBE_FACE + 9.0
    y += 1.0

    # table tags
    y = section(draw, y, "3. Table - IDs 10-13 (optional)",
                "For the mat corners: lets a script check the top camera later. Tape them flat.")
    x, cut, gap = MARGIN, TABLE_TAG + 8.0, 4.0
    for tag_id in TABLE_IDS:
        tag_block(page, draw, dictionary, tag_id, TABLE_TAG, x, y, cut)
        x += cut + gap

    draw.text((px(A4_MM[0] / 2), px(A4_MM[1] - 8)), "made by so101_real/make_aruco_sheet.py  -  page 1", fill=120,
              font=font(2.4), anchor="mm")
    return page


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(here, "aruco_tags_A4.pdf"))
    ap.add_argument("--png", help="also save the page as a PNG (preview)")
    args = ap.parse_args()
    pages = build()
    pages[0].save(args.out, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    if args.png:                                     # name.png, name_2.png, ...
        stem, ext = os.path.splitext(args.png)
        for i, p in enumerate(pages):
            p.save(args.png if i == 0 else f"{stem}_{i + 1}{ext}")
    print(f"written: {args.out}  ({len(pages)} A4 pages, {DPI} dpi) - print at 100 % / Actual size")


if __name__ == "__main__":
    main()
