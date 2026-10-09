"""LLM backends. Each exposes ``call(...) -> (raw_answer: dict, meta: dict)``; meta carries
``model``, ``turns``, ``cost_usd``, ``input_tokens`` and ``output_tokens`` when known."""
