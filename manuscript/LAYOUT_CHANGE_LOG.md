# LaTeX layout change log

The Elsevier `cas-sc` class, original fonts, page geometry, line spacing and numbered section hierarchy remain unchanged. No figure was replaced or redrawn; all9 actual image files have identical bytes to the baseline.

- Standard float spacing: text-to-float skip changed from12pt to10pt, with ordinary stretch/shrink; float-to-float skip changed from10pt to9pt. In-text spacing remains10pt.
- Standard counters allow three top floats, two bottom floats and five floats per page. This helps group compatible appendix tables without forcing repeated empty regions.
- Float-only pages are top-aligned, with12pt separation rather than vertically stretched gaps; remaining flexible space is placed below the content.
- The unnecessary `clearpage` before the appendix was replaced with a targeted `FloatBarrier`. Other section/bibliography guards remain, keeping HBAR evidence near its section and appendix floats out of references.
- Existing shrink-only table fitting remains: short tables are not enlarged and wide tables fit the text width. No global font or margin reduction was used.
- Paragraph consolidation and a concise conclusion improve natural text flow. The PDF decreased from33 pages (including highlights) to32; page-count reduction was an outcome, not a target.
- Highlights author punctuation was given explicit space without changing names, affiliations or roles.

The final32-page PDF was visually reviewed using the actual figures. All31 numbered article pages and the highlights page remain legible. The inherited CAS maketitle overfull-hbox warning and several underfull boxes persist in the compiler log without a visible overflow in the inspected output; no unsupported “zero warnings” claim is made.
