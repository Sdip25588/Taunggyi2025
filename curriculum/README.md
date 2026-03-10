# Curriculum PDFs

This folder holds the curriculum PDFs used by the RAG pipeline.

## Required Files

Place the following PDFs in this directory before running the app:

| Filename | Contents |
|----------|----------|
| `phonics.pdf` | Phonics curriculum for Grades 1–3. Covers letter sounds, blends, digraphs. |
| `reading.pdf` | McGuffey's First Eclectic Reader. Graded reading lessons from letters to short stories. |
| `Spelling.pdf` | McGuffey's Eclectic Spelling Book. Graded word lists with phonetic notation. |

## Notes

- All three PDFs are **text-based** (searchable text, not scanned images).
- They may also contain diagrams and tables that PyPDF2 will extract as text.
- The filenames are **case-sensitive** — use exactly `phonics.pdf`, `reading.pdf`, `Spelling.pdf`.

## After Adding PDFs

1. Start the app: `streamlit run main.py`
2. On first run, the app will automatically build a FAISS vector index from the PDFs.
3. The index is saved to `data/faiss_index/` so it only needs to be built once.
4. To rebuild the index (e.g., after updating PDFs), delete the `data/faiss_index/` folder and restart.

## Where to Get These PDFs

- **McGuffey's First Eclectic Reader** — Available at [Project Gutenberg](https://www.gutenberg.org/)
- **McGuffey's Eclectic Spelling Book** — Available at [Project Gutenberg](https://www.gutenberg.org/)
- **Phonics curriculum** — Contact your school library or curriculum provider.
