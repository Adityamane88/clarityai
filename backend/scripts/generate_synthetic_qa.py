from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import get_settings
from app.db.models import KnowledgeDocument
from app.db.session import SessionLocal

settings = get_settings()

PROMPT_TEMPLATE = '''You are building a supervised fine-tuning dataset for a grounded assistant.
Read the document chunk below and create {qa_count} high-quality user questions and assistant answers.
The questions should be realistic, practical, and varied. The answers should be concise but specific.
Return valid JSON only in this format:
[
  {{"question": "...", "answer": "..."}}
]

Document title: {title}
Chunk:
{content}
'''


def chat_completion(messages: list[dict]) -> str:
    if not settings.remote_llm_configured:
        raise RuntimeError('Set LLM_BASE_URL, LLM_API_KEY, and CHAT_MODEL before running this script.')
    url = f'{settings.llm_base_url}/chat/completions'
    headers = {'Content-Type': 'application/json'}
    if settings.llm_api_key:
        headers['Authorization'] = f'Bearer {settings.llm_api_key}'
    payload = {
        'model': settings.chat_model,
        'temperature': 0.3,
        'messages': messages,
        'stream': False,
    }
    timeout = httpx.Timeout(60.0, connect=20.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data['choices'][0]['message']['content']


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic QA pairs from uploaded knowledge documents.')
    parser.add_argument('--out', required=True, help='Output JSONL path')
    parser.add_argument('--limit-docs', type=int, default=20, help='Maximum number of documents to sample')
    parser.add_argument('--chunks-per-doc', type=int, default=3, help='Maximum number of chunks to sample per document')
    parser.add_argument('--qa-count', type=int, default=3, help='QA pairs to request per chunk')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    created = 0
    try:
        documents = db.execute(
            select(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .order_by(KnowledgeDocument.updated_at.desc())
            .limit(args.limit_docs)
        ).scalars().all()
        with out_path.open('w', encoding='utf-8') as handle:
            for document in documents:
                for chunk in list(document.chunks)[: args.chunks_per_doc]:
                    prompt = PROMPT_TEMPLATE.format(
                        qa_count=args.qa_count,
                        title=document.title,
                        content=chunk.content[:1800],
                    )
                    try:
                        content = chat_completion(
                            [
                                {'role': 'system', 'content': 'Return valid JSON only.'},
                                {'role': 'user', 'content': prompt},
                            ]
                        )
                        rows = json.loads(content)
                    except Exception as exc:
                        print(f'Skipping chunk {chunk.id}: {exc}')
                        continue
                    for row in rows:
                        question = (row.get('question') or '').strip()
                        answer = (row.get('answer') or '').strip()
                        if not question or not answer:
                            continue
                        payload = {
                            'messages': [
                                {'role': 'system', 'content': 'You are ClarityAI. Give grounded, practical answers.'},
                                {'role': 'user', 'content': question},
                                {'role': 'assistant', 'content': answer},
                            ],
                            'metadata': {
                                'document_id': document.id,
                                'document_title': document.title,
                                'chunk_id': chunk.id,
                            },
                        }
                        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
                        created += 1
    finally:
        db.close()

    print(f'Wrote {created} synthetic QA examples to {out_path}')


if __name__ == '__main__':
    main()
