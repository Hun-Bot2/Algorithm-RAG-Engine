# Optional local encoder adapter

This offline-only adapter consumes an already provisioned model directory with a
reviewed file-hash manifest. It never downloads models or calls an inference
server. Sentence Transformers is an optional research dependency, imported only
when explicitly loading the adapter. Core Python and Worker code never import it.

Each manifest pins an immutable 40-hex revision, model ID, license reference,
namespace, dimensions, max tokens, and hashes of every local file. Only the two
compact candidates are accepted initially. E5 uses query/passage prefixes; MiniLM
uses no prefix. Input consists of the same independently authored profile fields
as TF-IDF. Query and catalog use the same model namespace but distinct required
preprocessing roles. L2-normalized outputs become bounded cosine neighbor lists,
with input/config/ordered-key hashes. A different model requires a new namespace;
these vectors cannot be compared to legacy OpenAI vectors.

Sources checked 2026-10-16: official SentenceTransformer API reference
https://www.sbert.net/docs/package_reference/sentence_transformer/model.html and
model cards https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and
https://huggingface.co/intfloat/multilingual-e5-small. The API documents
local_files_only, revision, trust_remote_code and CPU device options. Tests use
an injected encoder to validate contracts; no real weights or transformer
runtime were installed or measured. Actual encoder quality/resources remain open.
