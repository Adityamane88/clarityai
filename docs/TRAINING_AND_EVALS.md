# Training and evaluation path

## What "training properly" means here

For an app like this, the best production path is not trillion-record pretraining.
It is a layered quality loop:

1. Use a strong base model.
2. Ground answers in your own documents.
3. Add web research for missing or time-sensitive knowledge.
4. Capture feedback from real users.
5. Fine-tune later on high-quality conversations.
6. Keep an eval set and test every release.

## Scripts included

### Export good answers

```bash
python backend/scripts/export_feedback_dataset.py --out data/positive_feedback.jsonl
```

### Build SFT dataset

```bash
python backend/scripts/build_sft_dataset.py --out data/clarity_sft.jsonl
```

### Generate synthetic QA from your uploaded knowledge

```bash
python backend/scripts/generate_synthetic_qa.py --out data/synthetic_qa.jsonl
```

This uses your configured chat model to create additional supervised examples.

### Run retrieval and routing evals

```bash
python backend/scripts/run_eval_suite.py --eval-set backend/evals/sample_eval_set.json
```

## Recommended release process

- Reindex knowledge.
- Run eval suite.
- Manually inspect a few answers.
- Deploy only if retrieval and routing pass rates are acceptable.
- Monitor user feedback after release.
