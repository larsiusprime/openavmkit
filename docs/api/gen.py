# docs/api/gen.py
from pathlib import Path
import pkgutil
import importlib
import inspect

import mkdocs_gen_files

package = "openavmkit"

def has_public_content(mod) -> bool:
    """
    Return True if the module itself defines at least one public (non-underscore)
    function or class.
    """
    for name, obj in inspect.getmembers(mod):
        # skip anything not a function/class
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        # ensure it's defined in this module, not imported
        if obj.__module__ != mod.__name__:
            continue
        # only count truly public names
        if not name.startswith("_"):
            return True
    return False


def docs_path_for(module_name: str) -> Path:
  parts = module_name.split(".")[1:] # strip leading "openavmkit"
  if len(parts) == 1:
    return Path("api", "Core", parts[0]).with_suffix(".md")
  return Path("api", *parts).with_suffix(".md")


for module in pkgutil.walk_packages([package.replace(".", "/")], prefix=f"{package}."):
    mod = importlib.import_module(module.name)
    if not has_public_content(mod):
      continue
    # Skip private or empty modules
    if module.name.endswith(".__main__"):
      continue
    doc_path = docs_path_for(module.name)
    with mkdocs_gen_files.open(doc_path.with_suffix(".md"), "w") as f:
      ident = module.name
      f.write(f"# `{ident}`\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(doc_path, Path(module.module_finder.path) / "__init__.py")

