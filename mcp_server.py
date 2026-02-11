#!/usr/bin/env python3
"""
DataWarrior-Agent MCP Server v1
Exposes DataWarrior tools via the MCP protocol (Model Context Protocol)

Features:
- read_dataset: Reading the dataset for LLM analysis
- Auto-save snapshot after each operation modifying data
- Token management and limits

Usage:
    python mcp_server.py                    # Runs in stdio mode (for MCP clients)
    python mcp_server.py --test             # Local tool testing
    python mcp_server.py --list             # Lists available tools
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.tcp_client import tcp_client, check_connection, send_macro_to_datawarrior
from core.macro_modifier import MacroModifier

# ============================================
# CONFIGURATION
# ============================================

PROJECT_DIR = Path(__file__).parent
MACROS_DIR = PROJECT_DIR / "macros"
CONFIG_DIR = PROJECT_DIR / "config"
DATA_INPUT_DIR = PROJECT_DIR / "data" / "input"
DATA_OUTPUT_DIR = PROJECT_DIR / "data" / "output"

# Path to the CSV snapshot (for read_dataset)
SNAPSHOT_CSV_PATH = DATA_OUTPUT_DIR / "_snapshot.csv"

# Create directories if needed
for dir_path in [MACROS_DIR, CONFIG_DIR, DATA_INPUT_DIR, DATA_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Load the descriptors catalog
DESCRIPTORS_CATALOG_PATH = CONFIG_DIR / "descriptors_catalog.json"

def load_descriptors_catalog() -> dict:
    """Loads the descriptors catalog"""
    if DESCRIPTORS_CATALOG_PATH.exists():
        with open(DESCRIPTORS_CATALOG_PATH, 'r') as f:
            return json.load(f)
    return {"all_descriptors": [], "categories": {}, "aliases": {}}

DESCRIPTORS_CATALOG = load_descriptors_catalog()

# Macro modifier instance
macro_modifier = MacroModifier(MACROS_DIR)

# ============================================
# SESSION STATE
# ============================================

class SessionState:
    """DataWarrior session state"""
    
    def __init__(self):
        self.source_file: Optional[str] = None
        self.smiles_column: Optional[str] = None
        self.structure_column: str = "Structure"
        self.file_loaded: bool = False
        self.history: List[dict] = []
    
    def set_source(self, filename: str, smiles_column: Optional[str] = None):
        """Configures the source file"""
        self.source_file = filename
        self.smiles_column = smiles_column
        self.structure_column = f"Structure of {smiles_column}" if smiles_column else "Structure"
        self.file_loaded = True
        
        # Update all macros with the correct structure column
        macro_modifier.update_all_structure_columns(self.structure_column)
    
    def add_to_history(self, action: str, details: dict):
        """Adds an action to the history"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        })
    
    def to_dict(self) -> dict:
        """Exports the state as dict"""
        return {
            "source_file": self.source_file,
            "smiles_column": self.smiles_column,
            "structure_column": self.structure_column,
            "file_loaded": self.file_loaded,
            "history_count": len(self.history)
        }

# Global instance
session = SessionState()

# ============================================
# MCP SERVER
# ============================================

mcp = FastMCP("DataWarrior-Agent")

# ============================================
# HELPER: AUTO-SAVE SNAPSHOT
# ============================================

def _auto_save_snapshot() -> bool:
    """
    Automatic dataset save after an operation.
    Used so that read_dataset can access current data.
    """
    try:
        # Update the macro to save to _snapshot.csv
        result = macro_modifier.update_filename_in_task(
            "save_csv_file",
            "saveCommaSeparatedFile",
            "./data/output/_snapshot.csv"
        )
        
        if result["status"] != "success":
            print(f"[Auto-save] Failed to update macro: {result['message']}")
            return False
        
        # Execute the macro
        macro_path = MACROS_DIR / "save_csv_file.dwam"
        result = send_macro_to_datawarrior(macro_path, wait=True)
        
        if result["status"] == "success":
            print("[Auto-save] Snapshot saved successfully")
            return True
        else:
            print(f"[Auto-save] Failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"[Auto-save] Error: {e}")
        return False

# ============================================
# TOOLS - Dataset Reading (NEW)
# ============================================

@mcp.tool()
def read_dataset() -> dict:
    """
    Read the current dataset to analyze its contents.
    
    This function reads the latest snapshot of the dataset, allowing analysis
    of the data (column names, values, statistics, filtering, etc.).
    
    If no snapshot exists but DataWarrior TCP is connected, it will automatically
    create one.
    
    Returns:
        dict with:
        - status: "success" | "warning" | "error"
        - message: Description of the result
        - data: CSV content as string (cleaned, without Structure columns)
        - metadata: {rows, columns, columns_count, tokens}
        
    Use this when the user asks about:
    - Column names in the dataset
    - Number of molecules/rows
    - Specific values or statistics
    - Data filtering questions (e.g., "molecules with logP > 3")
    - Any question requiring access to the actual data
    """
    import pandas as pd
    
    # If the snapshot doesn't exist, try to create it
    if not SNAPSHOT_CSV_PATH.exists():
        if check_connection():
            print("[read_dataset] No snapshot found, creating one...")
            _auto_save_snapshot()
            # Wait a bit for the file to be written
            import time
            time.sleep(1)
        else:
            return {
                "status": "error",
                "message": "No dataset snapshot available. DataWarrior TCP is not connected. Please start the TCP plugin (Tools → Start Macro Agent) and try again.",
                "data": "",
                "metadata": {}
            }
    
    # Check again after creation attempt
    if not SNAPSHOT_CSV_PATH.exists():
        return {
            "status": "error",
            "message": "Could not create dataset snapshot. Please try using 'save_as_csv' first.",
            "data": "",
            "metadata": {}
        }
    
    try:
        # Use pandas to parse correctly (automatic separator detection)
        # Try first with comma, then tab
        try:
            df = pd.read_csv(SNAPSHOT_CSV_PATH, sep=',', encoding='utf-8')
            # If only one column, probably wrong separator
            if len(df.columns) == 1:
                df = pd.read_csv(SNAPSHOT_CSV_PATH, sep='\t', encoding='utf-8')
        except:
            df = pd.read_csv(SNAPSHOT_CSV_PATH, sep='\t', encoding='utf-8')
        
        # Columns to exclude (encoded Structure, too long and useless for analysis)
        columns_to_exclude = []
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['structure', 'idcode', 'smiles']):
                columns_to_exclude.append(col)
        
        # Create a cleaned DataFrame
        df_clean = df.drop(columns=columns_to_exclude, errors='ignore')
        
        # Statistics
        num_rows = len(df_clean)
        columns = list(df_clean.columns)
        
        # Convert to clean CSV
        clean_csv = df_clean.to_csv(index=False)
        
        # Estimate tokens
        tokens = len(clean_csv) // 4
        
        # Also prepare a summary of excluded columns
        excluded_info = f" (excluded columns: {', '.join(columns_to_exclude)})" if columns_to_exclude else ""
        
        metadata = {
            "rows": num_rows,
            "columns": columns,
            "columns_count": len(columns),
            "tokens": tokens,
            "excluded_columns": columns_to_exclude,
            "file_path": str(SNAPSHOT_CSV_PATH)
        }
        
        # Check token limits and warn if large
        warning_threshold = 30000  # For large models
        
        if tokens > warning_threshold:
            return {
                "status": "warning",
                "message": f"⚠️ Large dataset ({tokens} tokens, {num_rows} molecules, {len(columns)} columns){excluded_info}. Analysis may be incomplete.",
                "data": clean_csv,
                "metadata": metadata
            }
        
        return {
            "status": "success",
            "message": f"Dataset loaded: {num_rows} molecules, {len(columns)} columns ({tokens} tokens){excluded_info}",
            "data": clean_csv,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading dataset: {str(e)}",
            "data": "",
            "metadata": {}
        }

# ============================================
# TOOLS - File Operations
# ============================================

@mcp.tool()
def open_file(filename: str) -> dict:
    """
    Open a chemical file in DataWarrior.
    
    This tool opens molecular data files (SDF, MOL, CSV with SMILES, etc.) 
    in DataWarrior and initializes the TCP macro agent.
    
    Args:
        filename: Path to the file to open (relative to data/input/ or absolute path)
        
    Returns:
        Status dict with success/error message
        
    Example:
        open_file("molecules.sdf")
        open_file("compounds.csv")
    """
    # Resolve the path
    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = DATA_INPUT_DIR / filename
    
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }
    
    # Copy to user_upload.{ext} to standardize
    ext = file_path.suffix.lower()
    dest_path = DATA_INPUT_DIR / f"user_upload{ext}"
    shutil.copy(file_path, dest_path)
    
    # Detect if it's a file with SMILES
    smiles_column = None
    if ext in ['.csv', '.txt', '.tsv']:
        smiles_column = _detect_smiles_column(file_path)
    
    # Update the open.dwam macro
    relative_path = f"./data/input/user_upload{ext}"
    result = macro_modifier.update_filename("open", relative_path)
    
    if result["status"] != "success":
        return result
    
    # Configure the session
    session.set_source(filename, smiles_column)
    
    # Launch DataWarrior with the open macro (via command line since TCP is not yet active)
    macro_path = MACROS_DIR / "open.dwam"
    
    try:
        # Launch DataWarrior in the VNC display
        cmd = [
            "env", "DISPLAY=:2",
            "/opt/datawarrior/datawarrior",
            str(macro_path.absolute())
        ]
        
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        session.add_to_history("open_file", {"filename": filename, "smiles_column": smiles_column})
        
        # Wait a bit and do an initial auto-save
        time.sleep(3)
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": f"Opening {filename} in DataWarrior...",
            "structure_column": session.structure_column,
            "smiles_column": smiles_column,
            "note": "Wait a few seconds for DataWarrior to start, then the TCP agent will be ready"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to launch DataWarrior: {str(e)}"
        }


@mcp.tool()
def save_as_csv(filename: str = "exported_data.csv") -> dict:
    """
    Save the current dataset as a CSV file.
    
    Args:
        filename: Output filename (will be saved in data/output/)
        
    Returns:
        Status dict with success/error message
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    # Update the path in the macro
    output_path = f"./data/output/{filename}"
    result = macro_modifier.update_filename_in_task(
        "save_csv_file", 
        "saveCommaSeparatedFile",
        output_path
    )
    
    if result["status"] != "success":
        return result
    
    # Send the macro
    macro_path = MACROS_DIR / "save_csv_file.dwam"
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("save_csv", {"filename": filename})
        
        # ALSO create the snapshot for read_dataset
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": f"Dataset saved to: data/output/{filename}"
        }
    
    return result


# ============================================
# TOOLS - Descriptor Calculation
# ============================================

@mcp.tool()
def calculate_all_descriptors() -> dict:
    """
    Calculate ALL available molecular descriptors.
    
    This calculates all 59 molecular descriptors including:
    - Weight properties (MW, fragment weights)
    - Drug-likeness (logP, logS, TPSA, HBD, HBA, etc.)
    - Toxicity predictions (mutagenic, tumorigenic, irritant, PAINS)
    - Structural properties (complexity, flexibility, shape)
    - Atom and ring counts
    - 3D properties (globularity, surface, volume)
    
    Returns:
        Status dict with success/error message
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    macro_path = MACROS_DIR / "calcul_all_descriptors.dwam"
    
    if not macro_path.exists():
        return {
            "status": "error",
            "message": f"Macro not found: {macro_path}"
        }
    
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("calculate_all_descriptors", {})
        
        # Auto-save snapshot
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": "Calculated all 59 molecular descriptors",
            "descriptors_count": 59
        }
    
    return result


@mcp.tool()
def calculate_descriptors(descriptors: List[str]) -> dict:
    """
    Calculate SPECIFIC molecular descriptors.
    
    Args:
        descriptors: List of descriptor names to calculate.
        
    Available descriptors by category:
    
    WEIGHT: totalWeight, fragmentWeight, fragmentAbsWeight
    
    DRUGLIKENESS: logP, logS, acceptors (HBA), donors (HBD), sasa, rpsa, tpsa, druglikeness
    
    TOXICITY: mutagenic, tumorigenic, reproEffective, irritant, nasty, pains
    
    STRUCTURE: shape, flexibility, complexity, fragments
    
    ATOMS: heavyAtoms, nonCHAtoms, metalAtoms, negAtoms, stereoCenters, 
           aromAtoms, sp3CFraction, sp3Atoms, symmetricAtoms
    
    BONDS: nonHBonds, rotBonds, closures
    
    RINGS: largestRing, rings, carbo, heteroRings, satRings, nonAromRings, 
           aromRings, carboSatRings, carboNonAromRings, carboAromRings,
           heteroSatRings, heteroNonAromRings, heteroAromRings
    
    FUNCTIONAL GROUPS: amides, amines, alkylAmines, arylAmines, aromN, basicN, acidicO
    
    STEREO: stereoConfiguration
    
    3D: globularity, globularity2, surface3d, volume3d
    
    Returns:
        Status dict with success/error message
        
    Example:
        calculate_descriptors(["logP", "logS", "tpsa", "donors", "acceptors"])
        calculate_descriptors(["totalWeight", "rotBonds", "rings"])
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    if not descriptors:
        return {
            "status": "error",
            "message": "No descriptors specified. Please provide a list of descriptor names."
        }

    # Validate and resolve aliases
    valid_descriptors = []
    invalid_descriptors = []
    all_valid = DESCRIPTORS_CATALOG.get("all_descriptors", [])
    aliases = DESCRIPTORS_CATALOG.get("aliases", {})

    print(f"[DEBUG] Input descriptors: {descriptors}")

    for desc in descriptors:
        # Try exact match first
        if desc in all_valid:
            valid_descriptors.append(desc)
        # Try aliases
        elif desc.lower() in aliases:
            resolved = aliases[desc.lower()]
            valid_descriptors.append(resolved)
        # Try case-insensitive
        else:
            found = False
            for valid in all_valid:
                if desc.lower() == valid.lower():
                    valid_descriptors.append(valid)
                    found = True
                    break
            if not found:
                invalid_descriptors.append(desc)

    if invalid_descriptors:
        return {
            "status": "error",
            "message": f"Invalid descriptor(s): {', '.join(invalid_descriptors)}. Use get_available_descriptors() to see valid names."
        }

    if not valid_descriptors:
        return {
            "status": "error",
            "message": "No valid descriptors to calculate."
        }

    # Update the macro with selected descriptors
    result = macro_modifier.update_property_list("calcul_selective_descriptors", valid_descriptors)
    
    if result["status"] != "success":
        return result
    
    # Send the macro
    macro_path = MACROS_DIR / "calcul_selective_descriptors.dwam"
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("calculate_descriptors", {"descriptors": valid_descriptors})
        
        # Auto-save snapshot
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": f"Calculated {len(valid_descriptors)} descriptor(s): {', '.join(valid_descriptors)}"
        }
    
    return result


@mcp.tool()
def get_available_descriptors(category: Optional[str] = None) -> dict:
    """
    List all available molecular descriptors.
    
    Args:
        category: Optional category filter. 
                  Options: weight, druglikeness, toxicity, structure, atoms, bonds, rings, functional_groups, stereo, 3d
        
    Returns:
        Dict with descriptors organized by category
    """
    categories = DESCRIPTORS_CATALOG.get("categories", {})
    
    if category:
        if category in categories:
            return {
                "status": "success",
                "category": category,
                "description": categories[category].get("description", ""),
                "descriptors": categories[category].get("descriptors", [])
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown category: {category}. Available: {', '.join(categories.keys())}"
            }
    
    return {
        "status": "success",
        "categories": categories,
        "total_count": len(DESCRIPTORS_CATALOG.get("all_descriptors", [])),
        "aliases": DESCRIPTORS_CATALOG.get("aliases", {})
    }


# ============================================
# TOOLS - Filters
# ============================================

@mcp.tool()
def apply_lipinski_filter() -> dict:
    """
    Apply Lipinski's Rule of Five filter.
    
    This filter removes molecules that violate drug-likeness criteria:
    - Molecular Weight ≤ 500 Da
    - LogP ≤ 5
    - Hydrogen Bond Donors ≤ 5
    - Hydrogen Bond Acceptors ≤ 10
    
    Molecules violating more than one rule are filtered out.
    
    Returns:
        Status dict with success/error message
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    macro_path = MACROS_DIR / "lipinski_filter.dwam"
    
    if not macro_path.exists():
        return {
            "status": "error",
            "message": f"Macro not found: {macro_path}"
        }
    
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("apply_lipinski_filter", {})
        
        # Auto-save snapshot
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": "Applied Lipinski Rule of Five filter. Non-druglike molecules have been filtered out."
        }
    
    return result


# ============================================
# TOOLS - Data Manipulation
# ============================================

@mcp.tool()
def delete_columns(columns: List[str]) -> dict:
    """
    Delete specified columns from the dataset.
    
    Args:
        columns: List of column names to delete
        
    Returns:
        Status dict with success/error message
        
    Example:
        delete_columns(["Molecule Name", "Comment"])
        delete_columns(["logP", "logS"])
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    if not columns:
        return {
            "status": "error",
            "message": "No columns specified. Please provide a list of column names to delete."
        }
    
    # Update the macro
    result = macro_modifier.update_column_list("delete_column", columns)
    
    if result["status"] != "success":
        return result
    
    # Send the macro
    macro_path = MACROS_DIR / "delete_column.dwam"
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("delete_columns", {"columns": columns})
        
        # Auto-save snapshot
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": f"Deleted {len(columns)} column(s): {', '.join(columns)}"
        }
    
    return result


# ============================================
# TOOLS - Visualization
# ============================================

@mcp.tool()
def generate_2d_coordinates() -> dict:
    """
    Generate or regenerate 2D coordinates for molecular structures.
    
    This tool creates clean 2D depictions of molecules, useful for:
    - Improving visual representation
    - Standardizing structure display
    - Preparing structures for reports
    
    Returns:
        Status dict with success/error message
    """
    if not check_connection():
        return {
            "status": "error",
            "message": "DataWarrior TCP plugin not connected. Start it via: Tools → Start Macro Agent"
        }
    
    macro_path = MACROS_DIR / "generate_new_2d_coord.dwam"
    
    if not macro_path.exists():
        return {
            "status": "error",
            "message": f"Macro not found: {macro_path}"
        }
    
    result = send_macro_to_datawarrior(macro_path)
    
    if result["status"] == "success":
        session.add_to_history("generate_2d_coordinates", {})
        
        # Auto-save snapshot
        _auto_save_snapshot()
        
        return {
            "status": "success",
            "message": "Generated new 2D coordinates for all structures"
        }
    
    return result


# ============================================
# TOOLS - Status & Info
# ============================================

@mcp.tool()
def get_connection_status() -> dict:
    """
    Check the connection status to DataWarrior's TCP plugin.
    
    Returns:
        Connection status and instructions if not connected
    """
    connected = check_connection()
    
    return {
        "status": "success",
        "connected": connected,
        "message": "DataWarrior TCP plugin is connected and ready" if connected else "Not connected - Start the plugin in DataWarrior: Tools → Start Macro Agent",
        "tcp_port": 5151
    }


@mcp.tool()
def get_session_info() -> dict:
    """
    Get information about the current session.
    
    Returns:
        Current session state including loaded file, structure column, and history
    """
    return {
        "status": "success",
        "session": session.to_dict(),
        "tcp_connected": check_connection(),
        "available_macros": macro_modifier.list_macros(),
        "snapshot_available": SNAPSHOT_CSV_PATH.exists()
    }


@mcp.tool()
def list_available_tools() -> dict:
    """
    List all available DataWarrior tools and their descriptions.
    
    Returns:
        Dict with all tools organized by category
    """
    return {
        "status": "success",
        "tools": {
            "file_operations": {
                "open_file": "Open a chemical file (SDF, MOL, CSV, etc.)",
                "save_as_csv": "Export dataset to CSV"
            },
            "data_access": {
                "read_dataset": "Read current dataset to analyze contents (columns, values, statistics)"
            },
            "descriptors": {
                "calculate_all_descriptors": "Calculate all 59 molecular descriptors",
                "calculate_descriptors": "Calculate specific descriptors",
                "get_available_descriptors": "List available descriptors by category"
            },
            "filters": {
                "apply_lipinski_filter": "Apply Lipinski Rule of Five"
            },
            "data_manipulation": {
                "delete_columns": "Delete columns from dataset"
            },
            "visualization": {
                "generate_2d_coordinates": "Generate 2D structure coordinates"
            },
            "status": {
                "get_connection_status": "Check TCP connection",
                "get_session_info": "Get current session info",
                "list_available_tools": "List all tools"
            }
        }
    }


# ============================================
# HELPER FUNCTIONS
# ============================================

def _detect_smiles_column(file_path: Path) -> Optional[str]:
    """Detects the SMILES column in a CSV/TSV file"""
    import csv
    
    try:
        ext = file_path.suffix.lower()
        if ext == '.csv':
            sep = ','
        elif ext in ['.tsv', '.tab']:
            sep = '\t'
        else:
            with open(file_path, 'r') as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                sep = sniffer.sniff(sample).delimiter
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
        
        smiles_patterns = ['smiles', 'smile', 'smi', 'canonical_smiles', 'isomeric_smiles']
        
        for header in headers:
            if header.lower() in smiles_patterns:
                return header
            for pattern in smiles_patterns:
                if pattern in header.lower():
                    return header
        
        return None
        
    except Exception:
        return None


# ============================================
# CLI
# ============================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DataWarrior-Agent MCP Server")
    parser.add_argument("--test", action="store_true", help="Test tools locally")
    parser.add_argument("--list", action="store_true", help="List available tools")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n🔬 DataWarrior-Agent MCP Server v1\n")
        print("=" * 50)
        result = list_available_tools()
        for category, tools in result["tools"].items():
            print(f"\n📁 {category.upper()}")
            for tool_name, description in tools.items():
                print(f"   • {tool_name}: {description}")
        print("\n" + "=" * 50)
        return
    
    if args.test:
        print("\n🧪 Testing DataWarrior-Agent MCP Server\n")
        print("=" * 50)
        
        print("\n1. Testing TCP connection...")
        result = get_connection_status()
        print(f"   Connected: {result['connected']}")
        
        print("\n2. Testing descriptors catalog...")
        result = get_available_descriptors()
        print(f"   Total: {result['total_count']}")
        
        print("\n3. Testing session info...")
        result = get_session_info()
        print(f"   Snapshot available: {result['snapshot_available']}")
        
        print("\n4. Testing read_dataset...")
        result = read_dataset()
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        
        print("\n" + "=" * 50)
        print("✅ Tests completed\n")
        return
    
    mcp.run()


if __name__ == "__main__":
    main()
