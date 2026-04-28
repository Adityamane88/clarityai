from app.services.chunker import chunk_sections


def test_chunk_sections_preserves_page_label() -> None:
    sections = [
        {'text': 'First paragraph. ' * 80, 'page_label': '1'},
        {'text': 'Second page text. ' * 60, 'page_label': '2'},
    ]
    chunks = chunk_sections(sections, max_chars=180, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].page_label == '1'
    assert chunks[-1].page_label == '2'
