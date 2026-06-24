---
title: Workflow Builder
---

# Simulation Workflow Builder

Assemble a ColliderML simulation pipeline stage by stage — **geometry →
generation → simulation → digitization → reconstruction** — then validate it
and submit it to the SaaS backend. Add or remove stages to build exactly the
workflow you want.

The **Copilot** (bottom-right, on every page) can read and edit this pipeline
for you: ask it to *"build a ttbar pu200 workflow and validate it"* and watch
the cards appear. On other pages the same copilot answers documentation
questions.

<WorkflowBuilder />

::: tip How submission works
**Submit** sends `{channel, events, pileup, seed}` to `POST /v1/simulate` on the
backend (`COLLIDERML_BACKEND`, default `http://localhost:8000`). The endpoint
verifies your HuggingFace token, then dispatches to Perlmutter via the
Superfacility API — or completes in mock mode (~2s) when SFAPI credentials
aren't set. Results land as a HuggingFace dataset you can pull with
`colliderml.load(...)`.
:::
