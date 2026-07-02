from agents.pre_processor import parse_document


def test_parse_document_missing_file_returns_empty_list():
    assert parse_document("does/not/exist.pdf") == []


def test_parse_document_unsupported_extension_returns_empty_list(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_text("hello", encoding="utf-8")
    assert parse_document(str(f)) == []


def test_parse_document_chunks_txt_content(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(
        "Quantitative Aptitude covers Number Systems, Algebra, and Geometry. " * 30,
        encoding="utf-8",
    )
    chunks = parse_document(str(f))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) <= 1000 for c in chunks)  # chunk_size used by parse_document


def test_parse_document_chunks_md_content(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Topic\n\n" + ("Some study notes here. " * 30), encoding="utf-8")
    chunks = parse_document(str(f))
    assert len(chunks) > 0
