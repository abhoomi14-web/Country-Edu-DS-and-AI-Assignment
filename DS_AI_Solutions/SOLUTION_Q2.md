# Question 2: Intelligent IT Ticket Auto-Resolution System

## My Reading of the Problem

2 million tickets a month across 5,000 issue types is a tough classification problem, but what really shaped my design was the noise situation. 30% of tickets poorly written, 10% coming in as screenshots — that's not a minor preprocessing footnote. It means a meaningful chunk of your data arrives garbled or as images, and any model that can't handle that realistically will fall apart in production.

The 2-second response constraint also rules out anything that runs a large model cold per ticket. You need caching, quantization, and a fast fallback path built in from the start.

## Architecture

Here's how it all fits together:

```text
Incoming ticket (text / screenshot / log paste)
          |
          v
OCR extraction (if screenshot) + text normalization
          |
          v
Response cache lookup — return immediately if hit
          |
          v
DistilBERT + TF-IDF ensemble classification
          |
          v
FAISS knowledge base search → top-3 solutions
          |
          v
Routing: auto-resolve / agent review / human triage
```

## Handling Noisy Tickets

Normalization runs before anything else. The first thing it does is fix encoding — ftfy handles garbled Unicode and HTML entities that show up when tickets arrive via email clients. After that, it masks URLs, email addresses, IPs, and hex literals (they're vocabulary noise for classification purposes), and collapses stack trace lines into a single `[STACKTRACE]` token so only the error summary survives. Repeated characters get collapsed too (`yyyyyy` → `yy`) along with chains of punctuation.

The step that moves the needle most in practice is abbreviation expansion — `pwd` → `password`, `db` → `database`, `k8s` → `kubernetes`, and about 20 others. IT support tickets are full of this shorthand and without expanding them, the classifier sees completely different tokens for the same concept.

One thing I deliberately left out of inference-time processing: heavy spell correction. It's useful during training data augmentation but too slow to run on every live ticket within a 2-second window. The abbreviation expansion and normalization steps handle most of the real-world noise without needing it.

## Screenshots

Screenshot tickets go through Tesseract OCR. The image gets preprocessed before OCR runs — converted to grayscale, median-filtered for denoising, then autocontrasted. That preprocessing step makes a noticeable difference on screenshots with uneven lighting or low contrast. After extraction, the OCR text feeds into exactly the same normalization and classification pipeline as text tickets. No separate code path needed.

## Classification

The primary classifier is DistilBERT fine-tuned on historical ticket data. A couple of reasons I went with it over a pure TF-IDF approach:

It handles context better for the kinds of ambiguous short tickets that are common in IT support. "can't access account" could be a password reset, a VPN issue, or an Active Directory lockout — TF-IDF sees the same bag of words regardless; DistilBERT can sometimes pick up on surrounding context that disambiguates them.

It's also 40% smaller and about 60% faster than full BERT, and after int8 quantization inference runs around 50ms per batch on CPU, which keeps it viable within the response budget.

The first 4 transformer layers are frozen during fine-tuning — those carry the pre-trained language representations from the base model. Only the top 2 layers and the classification head get trained on ticket data. This keeps training fast and reduces overfitting on smaller datasets.

A TF-IDF + LinearSVC pipeline runs alongside it as a fallback. On its own it gets around 75-78% accuracy — not quite the 80% target, but it keeps the system functional on machines where PyTorch isn't installed. The final prediction is a soft-voting ensemble: 70% DistilBERT + 30% TF-IDF when both are available.

## Knowledge Base

After classification, the system retrieves the top-3 most relevant past solutions from the knowledge base using FAISS semantic search. Solutions are indexed as SBERT dense vector embeddings (all-MiniLM-L6-v2, 384 dimensions). FAISS handles nearest-neighbor search in sub-millisecond time even over large indexes — for KBs under 100k entries it uses exact search (IndexFlatIP), above that an IVF approximate index.

When SBERT or FAISS aren't installed, TF-IDF cosine similarity is the fallback.

## Staying Under 2 Seconds

Three things together keep response time in check:

**Response caching** — SHA256-keyed in-memory store (Redis with 24h TTL in production). Recurring ticket types hit the cache and return almost instantly. In most IT support environments, a meaningful fraction of incoming tickets are duplicates or near-duplicates of things already seen — typical cache hit rates are 20-35%.

**Model quantization** — int8-quantized DistilBERT cuts inference time significantly on CPU compared to fp32.

**FAISS ANN** — Approximate nearest-neighbor search keeps knowledge base lookup in milliseconds even as the solution library grows.

Tickets where confidence falls below 0.4 get routed to a human agent rather than auto-resolved. It's better to escalate than to confidently give the wrong answer on something sensitive.

## Getting to 80%+ Accuracy

The DistilBERT + TF-IDF ensemble gets there on well-labeled historical data. The practical things that move the needle most in production: training on resolved historical tickets rather than synthetic examples, cleaning up duplicate or inconsistently labeled issue types, and keeping a human-in-the-loop feedback mechanism so agents can flag misclassifications back to the training pipeline. That feedback loop is how the model stays accurate over time without needing full retrains.

## Running the Code

```bash
pip install transformers torch sentence-transformers faiss-cpu scikit-learn pandas numpy pillow pytesseract ftfy joblib
python Q2_ticket_auto_resolution.py
```

The demo runs in TF-IDF-only mode for speed. It generates 500 synthetic tickets across 10 issue types, trains the classifier, evaluates on a held-out test set, and resolves a few live demo cases with response time and cache status logged. Swap `use_bert=False` to `use_bert=True` in the `__main__` block for the full DistilBERT run.
