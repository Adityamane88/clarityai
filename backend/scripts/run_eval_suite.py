from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.db.session import SessionLocal
from app.services.retrieval import retrieval_index
from app.services.routing import choose_route


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a lightweight retrieval and routing eval suite.')
    parser.add_argument('--eval-set', required=True, help='Path to eval JSON file')
    parser.add_argument('--out', help='Optional path to write results JSON')
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    cases = json.loads(eval_path.read_text(encoding='utf-8'))

    db = SessionLocal()
    try:
        retrieval_index.rebuild(db)
        results = []
        passed = 0
        for case in cases:
            search = retrieval_index.search(case['query'])
            decision = choose_route(
                user_message=case['query'],
                local_confidence=search['confidence'],
                local_hits=len(search['results']),
                research_mode=case.get('research_mode', 'auto'),
            )
            returned_titles = [item['document_title'] for item in search['results']]
            expected_titles = case.get('expected_doc_titles', [])
            title_match = all(any(expected in title for title in returned_titles) for expected in expected_titles)
            route_match = decision.route == case.get('expected_route', decision.route)
            case_passed = title_match and route_match
            if case_passed:
                passed += 1
            results.append(
                {
                    'name': case.get('name', case['query'][:50]),
                    'passed': case_passed,
                    'query': case['query'],
                    'returned_titles': returned_titles,
                    'expected_doc_titles': expected_titles,
                    'confidence': search['confidence'],
                    'route': decision.route,
                    'expected_route': case.get('expected_route'),
                }
            )
    finally:
        db.close()

    summary = {
        'total': len(cases),
        'passed': passed,
        'failed': len(cases) - passed,
        'pass_rate': round((passed / len(cases)) * 100, 2) if cases else 0.0,
        'results': results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
