# Product Handbook

## ClarityAI v2 goals

ClarityAI v2 is designed as a conversational knowledge assistant.

### Primary goals

- Hold a back-and-forth conversation instead of one-shot analysis.
- Answer using uploaded knowledge and cite what it used.
- Preserve recent context so follow-up questions feel natural.
- Keep the interface simple enough for non-technical users.
- Support fast iteration by allowing new documents to be uploaded without retraining a full model.

### Retrieval strategy

The first production version can use lexical ranking with a hybrid scoring strategy. A later version can add dense embeddings and a reranker.

### UX expectations

The interface should feel fast, readable, and calm. Source visibility is important because trust increases when users can inspect where an answer came from.
