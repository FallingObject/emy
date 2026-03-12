# Agentic RAG notes

An agentic RAG system can combine retrieval, routing, memory, and synthesis.

Persistent memory can be split into:
- episodic memory
- semantic memory
- reflective memory

A training-to-deployment path can start with labeled examples and later switch to deployment mode.

If the router is uncertain, a human can provide a label such as:
- smalltalk
- followup
- course_query
- research
- visual
- unknown

Deployment can optionally lock training so the memory becomes read-only.
