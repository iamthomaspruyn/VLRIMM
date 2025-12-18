"""
Convert a folder of PDFs into a lightweight structured dataset (text + cropped figures + tables).

Output layout (default):
  docling_output/<paper_stem>/
    text.txt
    images/*.png
    tables/*.md
    .processed

Notes
- This script is resumable via a `.processed` marker per paper.
- Image export uses PDF page rendering + provenance bbox cropping (Docling picture items often do not contain bitmap bytes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Papers") if Path("Papers").exists() else Path("papers"),
        help="Directory containing PDF files.",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("docling_output"), help="Output directory.")
    args = ap.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    converter = DocumentConverter()

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        paper_name = pdf_path.stem
        paper_out = args.output_dir / paper_name
        paper_out.mkdir(exist_ok=True)

        # ---- Skip if already processed ----
        # A paper is considered complete if either:
        # - a marker file exists (written after a successful run), OR
        # - expected output folders/files already exist (for older runs without marker)
        done_marker = paper_out / ".processed"
        text_out = paper_out / "text.txt"
        images_dir = paper_out / "images"
        tables_dir = paper_out / "tables"

        already_done = done_marker.exists() or (text_out.exists() and images_dir.exists() and tables_dir.exists())
        if already_done:
            print(f"Skipping (already processed): {pdf_path.name}")
            continue

        print(f"Processing: {pdf_path.name}")

        result = converter.convert(str(pdf_path))
        doc = result.document

        # ---- Save text ----
        (paper_out / "text.txt").write_text(doc.export_to_markdown(), encoding="utf-8")

        # ---- Save images ----
        images_dir.mkdir(exist_ok=True)

        # Image export: render PDF pages and crop using provenance bboxes.
        try:
            import pypdfium2 as pdfium

            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            rendered_pages = {}  # page_no -> (PIL.Image, page_w_pt, page_h_pt)

            for pic_i, pic in enumerate(getattr(doc, "pictures", []) or []):
                prov_list = getattr(pic, "prov", None) or []
                if not prov_list:
                    continue

                prov = prov_list[0]
                page_no = getattr(prov, "page_no", None)
                bbox = getattr(prov, "bbox", None)
                if page_no is None or bbox is None:
                    continue

                if page_no not in rendered_pages:
                    page = pdf_doc.get_page(page_no - 1)  # Docling is 1-based; pdfium is 0-based
                    page_w_pt, page_h_pt = page.get_size()
                    bitmap = page.render(scale=2)  # ~144 dpi
                    page_img = bitmap.to_pil()
                    rendered_pages[page_no] = (page_img, page_w_pt, page_h_pt)
                    try:
                        page.close()
                    except Exception:
                        pass

                page_img, page_w_pt, page_h_pt = rendered_pages[page_no]
                scale_x = page_img.width / page_w_pt
                scale_y = page_img.height / page_h_pt

                l = float(getattr(bbox, "l"))
                r = float(getattr(bbox, "r"))
                t = float(getattr(bbox, "t"))
                b = float(getattr(bbox, "b"))
                origin = str(getattr(bbox, "coord_origin", "BOTTOMLEFT"))

                if "BOTTOMLEFT" in origin:
                    left_px = int(l * scale_x)
                    right_px = int(r * scale_x)
                    top_px = int((page_h_pt - t) * scale_y)
                    bottom_px = int((page_h_pt - b) * scale_y)
                else:
                    left_px = int(l * scale_x)
                    right_px = int(r * scale_x)
                    top_px = int(t * scale_y)
                    bottom_px = int(b * scale_y)

                left_px = max(0, min(left_px, page_img.width - 1))
                right_px = max(left_px + 1, min(right_px, page_img.width))
                top_px = max(0, min(top_px, page_img.height - 1))
                bottom_px = max(top_px + 1, min(bottom_px, page_img.height))

                crop = page_img.crop((left_px, top_px, right_px, bottom_px))
                crop.save(images_dir / f"picture_p{page_no}_{pic_i}.png")

            try:
                pdf_doc.close()
            except Exception:
                pass
        except Exception:
            # If rendering/cropping fails, continue without images.
            pass

        # ---- Save tables (markdown) ----
        tables_dir.mkdir(exist_ok=True)
        for i, table in enumerate(getattr(doc, "tables", []) or []):
            p = tables_dir / f"table_{i}.md"
            if hasattr(table, "export_to_markdown"):
                p.write_text(table.export_to_markdown(), encoding="utf-8")
            else:
                p.write_text(str(table), encoding="utf-8")

        done_marker.write_text("ok\n", encoding="utf-8")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
