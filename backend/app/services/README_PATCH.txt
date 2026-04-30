Replace these files inside backend/app/services:
- prompts.py
- routing.py
- retrieval.py
- chat_engine.py
- documents.py
- web_research.py

What changed:
- removes ugly inline [S1]/[W1] from answer body
- returns cleaner source metadata for the UI
- reduces generic answers by simplifying the prompt
- reduces overstuffed history/context
- strengthens retrieval scoring and weak-evidence filtering
- improves route decisions for casual chat vs research vs hybrid
- makes fallback answers cleaner and more product-like
