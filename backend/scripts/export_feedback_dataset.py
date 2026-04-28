from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import ChatSession
from app.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description='Export positively rated assistant turns into a JSONL dataset.')
    parser.add_argument('--out', required=True, help='Path to the output JSONL file')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    count = 0
    try:
        sessions = db.execute(select(ChatSession).options(selectinload(ChatSession.messages))).scalars().all()
        with out_path.open('w', encoding='utf-8') as handle:
            for session in sessions:
                messages = list(session.messages)
                for index, message in enumerate(messages):
                    if message.role != 'assistant' or message.feedback_rating != 'up':
                        continue
                    history = messages[max(0, index - 6): index + 1]
                    payload = {
                        'session_id': session.id,
                        'messages': [{'role': item.role, 'content': item.content} for item in history],
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    count += 1
    finally:
        db.close()

    print(f'Wrote {count} training examples to {out_path}')


if __name__ == '__main__':
    main()
