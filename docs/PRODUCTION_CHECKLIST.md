# ClarityAI production checklist

This app is production-oriented, but strong results come from deployment discipline as much as code.

## Model strategy

Do not train a model from scratch for this app.
Use this path instead:

1. Strong chat model for reasoning and tone.
2. Embeddings model for dense retrieval.
3. Retrieval over your private documents.
4. Web research only when needed.
5. Feedback capture, evals, and later fine-tuning.

## Infrastructure

- Use PostgreSQL instead of SQLite in production.
- Put the API behind HTTPS and a reverse proxy.
- Store uploads on persistent disk or object storage.
- Set rate limits and request size limits.
- Back up the database and uploaded files.
- Add monitoring for latency, errors, token usage, and feedback score.

## Safety and trust

- Keep the safety layer enabled.
- Show citations whenever the answer depends on a source.
- Log which route was used: local, research, or hybrid.
- Add human review for high-stakes domains.

## Quality loop

- Export good conversations into an SFT dataset.
- Generate synthetic QA pairs from high-value documents.
- Run the eval suite before each release.
- Track answer helpfulness from thumbs-up and thumbs-down signals.
