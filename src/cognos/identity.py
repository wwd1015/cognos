"""Who is at the keyboard.

COGNOS runs locally, so the person is whoever started this process: ``COGNOS_USER`` when set,
else the operating-system login. The name is recorded on decisions and answers so a team that
shares a runs folder can see who did what. It is a label, not authentication: anyone who can
write to the runs folder can set it.
"""

from __future__ import annotations

import getpass
import os
import socket


def whoami() -> str:
    name = (os.environ.get("COGNOS_USER") or "").strip()
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:  # no login name (a bare container): still a usable label
        return "unknown"


def host() -> str:
    try:
        return socket.gethostname() or "unknown-host"
    except Exception:
        return "unknown-host"


def stamp() -> str:
    """The person and the machine, for work a process is doing (``alan@laptop``)."""
    return f"{whoami()}@{host()}"
