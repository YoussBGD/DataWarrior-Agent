#!/usr/bin/env python3
"""
Macro Modifier for DataWarrior
Modifies .dwam files before execution
"""

import re
from pathlib import Path
from typing import List, Optional


class MacroModifier:
    """Modifies DataWarrior macros (.dwam)"""
    
    def __init__(self, macros_dir: Path):
        self.macros_dir = macros_dir
    
    def get_macro_path(self, macro_name: str) -> Path:
        """Returns the path of a macro"""
        return self.macros_dir / f"{macro_name}.dwam"
    
    def macro_exists(self, macro_name: str) -> bool:
        """Checks if a macro exists"""
        return self.get_macro_path(macro_name).exists()
    
    def read_macro(self, macro_name: str) -> Optional[str]:
        """Reads the content of a macro"""
        path = self.get_macro_path(macro_name)
        if not path.exists():
            return None
        return path.read_text(encoding='utf-8')
    
    def write_macro(self, macro_name: str, content: str) -> bool:
        """Writes the content of a macro"""
        try:
            path = self.get_macro_path(macro_name)
            path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False
    
    # ============================================
    # SPECIFIC MODIFICATIONS
    # ============================================
    
    def update_filename(self, macro_name: str, new_filename: str) -> dict:
        """
        Updates ALL fileNames in a macro (open, etc.)
        
        ⚠️ Warning: replaces ALL fileName=
        For macros with multiple fileName, use update_filename_in_task()
        
        Args:
            macro_name: Name of the macro (without .dwam)
            new_filename: New file path
            
        Returns:
            dict with status and message
        """
        content = self.read_macro(macro_name)
        if content is None:
            return {"status": "error", "message": f"Macro not found: {macro_name}"}
        
        # Pattern to find fileName=...
        pattern = r'fileName=[^\n<]+'
        
        if not re.search(pattern, content):
            return {"status": "error", "message": "No fileName parameter found in macro"}
        
        # Replace ALL fileNames
        updated = re.sub(pattern, f'fileName={new_filename}', content)
        
        if self.write_macro(macro_name, updated):
            return {"status": "success", "message": f"Updated fileName to: {new_filename}"}
        else:
            return {"status": "error", "message": "Failed to write macro"}
    
    def update_filename_in_task(self, macro_name: str, task_name: str, new_filename: str) -> dict:
        """
        Updates the fileName ONLY in a specific task.
        
        Useful when a macro has multiple fileName= (e.g.: save_csv_file with 
        saveCommaSeparatedFile + saveFileAs for update_file.dwar)
        
        Args:
            macro_name: Name of the macro (without .dwam)
            task_name: Name of the task to modify (e.g.: "saveCommaSeparatedFile")
            new_filename: New file path
            
        Returns:
            dict with status and message
            
        Example:
            update_filename_in_task("save_csv_file", "saveCommaSeparatedFile", "./data/output/test.csv")
        """
        content = self.read_macro(macro_name)
        if content is None:
            return {"status": "error", "message": f"Macro not found: {macro_name}"}
        
        # Pattern to find fileName= in a specific task
        # Captures: <task name="taskName">\nfileName=VALUE
        pattern = rf'(<task name="{task_name}">\s*fileName=)[^\n<]+'
        
        if not re.search(pattern, content):
            return {
                "status": "error", 
                "message": f"No fileName in task '{task_name}' found in macro"
            }
        
        # Replace only in this task
        updated = re.sub(pattern, rf'\g<1>{new_filename}', content)
        
        if self.write_macro(macro_name, updated):
            return {
                "status": "success", 
                "message": f"Updated fileName in task '{task_name}' to: {new_filename}"
            }
        else:
            return {"status": "error", "message": "Failed to write macro"}
    
    def update_property_list(self, macro_name: str, properties: List[str]) -> dict:
        """
        Updates the propertyList in the macro (calcul_selective_descriptors)
        
        Args:
            macro_name: Name of the macro
            properties: List of descriptors to calculate
            
        Returns:
            dict with status and message
        """
        content = self.read_macro(macro_name)
        if content is None:
            return {"status": "error", "message": f"Macro not found: {macro_name}"}
        
        # Pattern to find propertyList=...
        pattern = r'propertyList=[^\n<]+'
        
        if not re.search(pattern, content):
            return {"status": "error", "message": "No propertyList parameter found in macro"}
        
        # Create the new list with TAB as separator (DataWarrior format)
        property_list_str = "\t".join(properties)
        
        # Replace
        updated = re.sub(pattern, f'propertyList={property_list_str}', content)
        
        if self.write_macro(macro_name, updated):
            return {
                "status": "success", 
                "message": f"Updated propertyList with {len(properties)} descriptors"
            }
        else:
            return {"status": "error", "message": "Failed to write macro"}
    
    def update_column_list(self, macro_name: str, columns: List[str]) -> dict:
        """
        Updates the columnList in a macro (delete_column)
        
        Args:
            macro_name: Name of the macro
            columns: List of columns
            
        Returns:
            dict with status and message
        """
        content = self.read_macro(macro_name)
        if content is None:
            return {"status": "error", "message": f"Macro not found: {macro_name}"}
        
        # Pattern to find columnList=...
        pattern = r'columnList=[^\n<]+'
        
        if not re.search(pattern, content):
            return {"status": "error", "message": "No columnList parameter found in macro"}
        
        # Create the new list with TAB as separator
        column_list_str = "\t".join(columns)
        
        # Replace
        updated = re.sub(pattern, f'columnList={column_list_str}', content)
        
        if self.write_macro(macro_name, updated):
            return {
                "status": "success", 
                "message": f"Updated columnList: {', '.join(columns)}"
            }
        else:
            return {"status": "error", "message": "Failed to write macro"}
    
    def update_structure_column(self, macro_name: str, structure_column: str) -> dict:
        """
        Updates the structureColumn in a macro
        
        Args:
            macro_name: Name of the macro
            structure_column: Name of the structure column (e.g.: "Structure" or "Structure of SMILES")
            
        Returns:
            dict with status and message
        """
        content = self.read_macro(macro_name)
        if content is None:
            return {"status": "error", "message": f"Macro not found: {macro_name}"}
        
        # Pattern to find structureColumn=...
        pattern = r'structureColumn=[^\n<]+'
        
        if not re.search(pattern, content):
            return {"status": "error", "message": "No structureColumn parameter found in macro"}
        
        # Replace
        updated = re.sub(pattern, f'structureColumn={structure_column}', content)
        
        if self.write_macro(macro_name, updated):
            return {
                "status": "success", 
                "message": f"Updated structureColumn to: {structure_column}"
            }
        else:
            return {"status": "error", "message": "Failed to write macro"}
    
    def update_all_structure_columns(self, structure_column: str) -> dict:
        """
        Updates structureColumn in ALL macros that use it
        
        Args:
            structure_column: Name of the structure column
            
        Returns:
            dict with status and results
        """
        results = []
        macros_with_structure = [
            "calcul_all_descriptors",
            "calcul_selective_descriptors", 
            "lipinski_filter",
            "generate_new_2d_coord"
        ]
        
        for macro_name in macros_with_structure:
            if self.macro_exists(macro_name):
                result = self.update_structure_column(macro_name, structure_column)
                results.append({
                    "macro": macro_name,
                    "status": result["status"]
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success" if success_count > 0 else "error",
            "message": f"Updated {success_count}/{len(results)} macros",
            "details": results
        }
    
    def list_macros(self) -> List[str]:
        """Lists all available macros"""
        return [f.stem for f in self.macros_dir.glob("*.dwam")]
    
    def get_macro_info(self, macro_name: str) -> dict:
        """Returns information about a macro"""
        content = self.read_macro(macro_name)
        if content is None:
            return {"exists": False, "error": "Macro not found"}
        
        info = {
            "exists": True,
            "name": macro_name,
            "path": str(self.get_macro_path(macro_name)),
            "has_propertyList": "propertyList=" in content,
            "has_columnList": "columnList=" in content,
            "has_structureColumn": "structureColumn=" in content,
            "has_fileName": "fileName=" in content
        }
        
        # Extract current values
        if info["has_structureColumn"]:
            match = re.search(r'structureColumn=([^\n<]+)', content)
            if match:
                info["structureColumn"] = match.group(1).strip()
        
        if info["has_fileName"]:
            match = re.search(r'fileName=([^\n<]+)', content)
            if match:
                info["fileName"] = match.group(1).strip()
        
        return info
