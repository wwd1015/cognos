"""Exploratory analysis: the tools the Data Analyst may call and the Python it may write.

The agent never touches the data. It *requests* an analysis (a registered tool with parameters,
or a script); the engine runs it, keeps the result and the code as artifacts, and hands the
agent a small summary. ``tools`` are the built-ins, ``sandbox`` runs agent-written code in a
restricted subprocess, ``charts`` is the chart format both produce, and ``run`` is the executor.
"""
