"""
Core modules for DataWarrior-AI
"""

from .tcp_client import (
    DataWarriorTCPClient,
    TCPConfig,
    tcp_client,
    send_macro_to_datawarrior,
    check_connection
)

from .macro_modifier import MacroModifier

__all__ = [
    "DataWarriorTCPClient",
    "TCPConfig", 
    "tcp_client",
    "send_macro_to_datawarrior",
    "check_connection",
    "MacroModifier"
]
