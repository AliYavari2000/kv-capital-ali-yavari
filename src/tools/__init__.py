"""Agent tools, grouped per node.

Each node that is an LLM agent gets a toolkit module here (e.g.
``intake_tools`` for Node 1). A toolkit bundles the callable tools, their
OpenAI function schemas, and a dispatch table the agent loop in ``src.llm``
drives.
"""
