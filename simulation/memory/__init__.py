import importlib
import os

module_names = [
    f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith("_memory.py")
]

__all__ = []

for module_name in module_names:
    module = importlib.import_module(f".{module_name}", package=__name__)
    globals().update(
        {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }
    )
    __all__.extend([name for name in dir(module) if not name.startswith("_")])
