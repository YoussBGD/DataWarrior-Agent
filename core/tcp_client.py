#!/usr/bin/env python3
"""
TCP Client to communicate with the DataWarrior plugin
"""

import socket
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class TCPConfig:
    host: str = "127.0.0.1"
    port: int = 5151
    timeout: int = 10


class DataWarriorTCPClient:
    def __init__(self, config: Optional[TCPConfig] = None, project_dir: Path = None):
        self.config = config or TCPConfig()
        self.project_dir = project_dir or Path(__file__).parent.parent
        self.watch_file = self.project_dir / "data" / "input" / "update_file.dwar"
    
    def is_connected(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((self.config.host, self.config.port))
            return True
        except:
            return False
    
    def send_macro(self, macro_path: Path, wait_for_completion: bool = True, max_wait: int = 60) -> dict:
        """
        Sends a macro and waits for it to complete
        
        Args:
            macro_path: Path to the macro
            wait_for_completion: Wait for the macro to finish
            max_wait: Maximum wait time in seconds
        """
        print(f"[TCP DEBUG] Sending macro: {macro_path}")
        
        if not macro_path.exists():
            print(f"[TCP DEBUG] File not found!")
            return {"status": "error", "message": f"Macro not found: {macro_path}"}
        
        # Get the timestamp before execution
        old_mtime = None
        if wait_for_completion and self.watch_file.exists():
            old_mtime = self.watch_file.stat().st_mtime
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.config.timeout)
                s.connect((self.config.host, self.config.port))
                message = f"{macro_path.absolute()}\n"
                s.sendall(message.encode('utf-8'))
            print(f"[TCP DEBUG] Sent OK!")
        except ConnectionRefusedError:
            return {"status": "error", "message": "TCP plugin not running"}
        except Exception as e:
            print(f"[TCP DEBUG] Error: {e}")
            return {"status": "error", "message": f"TCP error: {str(e)}"}
        
        # Wait for the macro to finish (file modified)
        if wait_for_completion:
            print(f"[TCP DEBUG] Waiting for macro to complete...")
            completed = self._wait_for_file_change(old_mtime, max_wait)
            if completed:
                print(f"[TCP DEBUG] Macro completed!")
            else:
                print(f"[TCP DEBUG] Timeout waiting for macro")
                return {"status": "warning", "message": f"Macro sent but completion not confirmed"}
        
        return {"status": "success", "message": f"Macro executed: {macro_path.name}"}
    
    def _wait_for_file_change(self, old_mtime: Optional[float], max_wait: int) -> bool:
        """Waits for the file to be modified"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.watch_file.exists():
                new_mtime = self.watch_file.stat().st_mtime
                
                # If no previous mtime, or if the file has changed
                if old_mtime is None or new_mtime > old_mtime:
                    # Wait a bit more to ensure writing is complete
                    time.sleep(0.5)
                    return True
            
            time.sleep(0.5)
        
        return False


tcp_client = DataWarriorTCPClient()

def send_macro_to_datawarrior(macro_path: Path, wait: bool = True) -> dict:
    return tcp_client.send_macro(macro_path, wait_for_completion=wait)

def check_connection() -> bool:
    return tcp_client.is_connected()
