import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

tree = parser.parse(b"def foo():\n    return 1")
print(dir(tree.root_node))
print(tree.root_node.type)
