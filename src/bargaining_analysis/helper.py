# Precompile the pattern to find any \roever{var}{old_value}
import re
from pathlib import Path




def inject_values(tex_path: Path, **variables):
    """
    Reads a .tex file, finds all \roever{var}{...} placeholders,
    replaces the ... with the provided variables[var], and writes back.
    
    Parameters:
    - tex_path: pathlib.Path to the .tex file
    - variables: kwargs mapping var names to their replacement values
    
    Raises:
    - KeyError: if a var placeholder isn't found or if any remain afterward
    """
    path = Path(tex_path)
    content = path.read_text(encoding='utf-8')
    ROEVER_PATTERN = re.compile(r'\\roever\{(?P<var>\w+)\}\{[^}]*\}')
    
    # Replace each variable's placeholder via regex substitution
    for var, val in variables.items():
        pattern = re.compile(rf'\\roever\{{{var}\}}\{{[^}}]*\}}')
        replacement = rf'\\roever{{{var}}}{{{val}}}'
        content, count = pattern.subn(replacement, content)
        if count == 0:
            raise KeyError(f"No placeholder \\roever{{{var}}}{{...}} found in {tex_path}")
    
    # Check for any unreplaced placeholders
    #leftovers = ROEVER_PATTERN.findall(content)
    #if leftovers:
    #    raise KeyError(f"Unreplaced placeholders remain for variables: {set(leftovers)}")
    
    # Write the updated content back to the file
    path.write_text(content, encoding='utf-8')