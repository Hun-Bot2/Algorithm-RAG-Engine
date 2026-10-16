# ADR 001: deterministic hosted core, optional offline AI

Status: accepted for implementation, 2026-10-16.

Decision: use Worker/static assets + D1 + Access for the eventual private service.
Keep local SQLite for development/recovery and Python for offline profiling,
permitted ingestion and evaluation. Core policy and templates require no AI.
Preserve legacy OpenAI batch behavior in its separate compatibility boundary.

Choose existing metadata/reviewed relationships as the MVP. Existing OpenAI vectors
may supply optional known-catalog neighbors after rights review, in an immutable
namespace; they cannot encode new text. Evaluate compact open-source batch encoders
before adoption. Own-model training is gated, beginning with a small feature scorer.
No serving server is needed. Optional model failure removes only that signal.

Rejected for the MVP: mandatory paid APIs, always-on workstation, transformer
inference in Workers, training from historical proposals, symmetric embeddings as
prerequisite direction, and model-generated mastery. Static frontend + Supabase
plus scheduled Python remains an alternative but adds boundaries and scheduling
and inactivity concerns. Local-only cannot satisfy unattended phone delivery.

Consequences: more explicit curation and deterministic policy tests; no recurring
AI requirement. Infrastructure free-tier feasibility, data rights, human labels,
hosted security and real delivery remain unverified release gates. No external
service activation follows from this ADR. Model/license documentation referenced
in the user plan must be refreshed before optional model adoption.
