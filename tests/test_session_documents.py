from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestrator import main
from orchestrator.nodes import chat_response
from orchestrator.session_documents import build_document_context, process_upload_file
from tests.conftest import FAKE_CLIENT_ID, FAKE_USER_ID


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


def _docx_bytes(text: str) -> bytes:
    from docx import Document

    buffer = BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


def _pdf_bytes(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    buffer = BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 220 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = stream
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_process_upload_file_extracts_text_metadata() -> None:
    processed = await process_upload_file(
        _upload("brief.md", b"# Launch Brief\nUse customer proof and avoid hype.")
    )

    assert processed.filename == "brief.md"
    assert processed.byte_size > 0
    assert processed.char_count == len(processed.content_text)
    assert processed.summary.startswith("# Launch Brief")
    assert processed.sha256
    assert processed.metadata["extension"] == ".md"
    assert processed.metadata["document_kind"] == "text"


@pytest.mark.asyncio
async def test_process_upload_file_extracts_docx_text() -> None:
    processed = await process_upload_file(
        _upload("brief.docx", _docx_bytes("DOCX customer proof angle"))
    )

    assert "DOCX customer proof angle" in processed.content_text
    assert processed.metadata["extension"] == ".docx"
    assert processed.metadata["document_kind"] == "docx"
    assert processed.metadata["extraction"] == "python-docx"


@pytest.mark.asyncio
async def test_process_upload_file_extracts_pdf_text() -> None:
    processed = await process_upload_file(
        _upload("brief.pdf", _pdf_bytes("PDF customer proof angle"))
    )

    assert "PDF customer proof angle" in processed.content_text
    assert processed.metadata["extension"] == ".pdf"
    assert processed.metadata["document_kind"] == "pdf"
    assert processed.metadata["extraction"] == "pypdf"


def test_build_document_context_formats_uploaded_docs() -> None:
    context = build_document_context(
        [
            {
                "filename": "brief.md",
                "content_text": "Prioritize customer proof and retention language.",
            }
        ],
        max_chars=500,
    )

    assert context is not None
    assert "Uploaded Document 1: brief.md" in context
    assert "customer proof" in context


@pytest.mark.asyncio
async def test_chat_response_includes_session_document_context(make_runtime: Any) -> None:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Use the proof angle."))
    runtime = make_runtime(llm)
    state = {
        "messages": [HumanMessage(content="What should I remember from the doc?")],
        "session_id": "sess-doc-chat",
        "session_documents": [
            {
                "filename": "brief.md",
                "content_text": "Prioritize customer proof and retention language.",
            }
        ],
    }

    await chat_response.run(
        state,  # type: ignore[arg-type]
        {"configurable": {"runtime": runtime}},
    )

    sent_messages = llm.ainvoke.await_args.args[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert "Uploaded Document 1: brief.md" in sent_messages[0].content


@pytest.mark.asyncio
async def test_upload_session_document_processes_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"authorization": "Bearer token"},
    )
    get_or_create = AsyncMock()
    create_document = AsyncMock(
        return_value={
            "doc_id": "doc-123",
            "status": "ready",
            "created_at": "2026-06-21T00:00:00Z",
        }
    )

    monkeypatch.setattr(main, "extract_bearer", lambda _request: "token")
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value=FAKE_USER_ID))
    monkeypatch.setattr(main, "get_client_id", AsyncMock(return_value=FAKE_CLIENT_ID))
    monkeypatch.setattr(main.session_manager, "get_or_create_session", get_or_create)
    monkeypatch.setattr(main.session_documents, "create_session_document", create_document)

    response = await main.upload_session_document(
        "sess-upload",
        request,  # type: ignore[arg-type]
        _upload("brief.txt", b"Remember the uploaded customer story."),
    )

    assert response["doc_id"] == "doc-123"
    assert response["filename"] == "brief.txt"
    get_or_create.assert_awaited_once_with(
        svc,
        session_id="sess-upload",
        user_id=FAKE_USER_ID,
        client_id=FAKE_CLIENT_ID,
    )
    create_document.assert_awaited_once()
    persisted_doc = create_document.await_args.kwargs["document"]
    assert persisted_doc.content_text == "Remember the uploaded customer story."
