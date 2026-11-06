"""Validate notebook structure and Python syntax, without running heavy labs."""
import ast
from pathlib import Path
import nbformat

for path in sorted(Path('notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells, 1):
        if cell.cell_type == 'code':
            code = '\n'.join('' if line.lstrip().startswith(('!', '%')) else line
                             for line in cell.source.splitlines())
            try:
                ast.parse(code)
            except SyntaxError as error:
                raise SyntaxError(f'{path}, cell {index}: {error}') from error
    print(f'OK {path}')
