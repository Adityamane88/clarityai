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

DEFAULT_SYSTEM_PROMPT = (
    'You are ClarityAI. Give grounded, practical, problem-solving answers. '
    'Use citations when evidence is provided and be honest about uncertainty.'
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build an SFT-ready JSONL dataset from positively rated chats.')
    parser.add_argument('--out', required=True, help='Output JSONL path')
    parser.add_argument('--max-history', type=int, default=8, help='How many turns of history to keep before each assistant answer')
    parser.add_argument('--system-prompt', default=DEFAULT_SYSTEM_PROMPT, help='System prompt to prepend to each example')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    examples = 0
    try:
        sessions = db.execute(select(ChatSession).options(selectinload(ChatSession.messages))).scalars().all()
        with out_path.open('w', encoding='utf-8') as handle:
            for session in sessions:
                messages = list(session.messages)
                for index, message in enumerate(messages):
                    if message.role != 'assistant' or message.feedback_rating != 'up':
                        continue
                    history = messages[max(0, index - args.max_history): index + 1]
                    payload = {
                        'messages': [
                            {'role': 'system', 'content': args.system_prompt},
                            *[
                                {'role': item.role, 'content': item.content}
                                for item in history
                            ],
                        ],
                        'metadata': {
                            'session_id': session.id,
                            'assistant_message_id': message.id,
                            'title': session.title,
                        },
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    examples += 1
    finally:
        db.close()

    print(f'Wrote {examples} SFT examples to {out_path}')


if __name__ == '__main__':
    main()
