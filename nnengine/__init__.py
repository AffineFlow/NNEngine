import platform
import subprocess

def _detect_x86_microarch():
    """Returns the highest supported x86 microarchitecture."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = f.read().lower()
            if "avx512f" in cpu_info: return "avx512"
            if "avx2" in cpu_info: return "avx2"
            
        elif system == "Darwin": # Intel Mac
            sysctl = subprocess.check_output(["sysctl", "-a"], text=True).lower()
            if "hw.optional.avx512f: 1" in sysctl: return "avx512"
            if "hw.optional.avx2_0: 1" in sysctl: return "avx2"
    except Exception:
        pass
        
    return "generic"

# 1. Determine architecture and highest support
_machine = platform.machine().lower()
_arch = "generic"

if _machine in ("x86_64", "amd64"):
    _arch = _detect_x86_microarch()

# 2. Dynamic cascading import with safe fallbacks
_backend = None

if _arch == "avx512":
    try:
        from . import _backend_avx512 as _backend
    except ImportError:
        _arch = "avx2"

if _arch == "avx2":
    try:
        from . import _backend_avx2 as _backend
    except ImportError:
        _arch = "generic"

if _arch == "generic":
    from . import _backend_generic as _backend

# 3. Automatically pull everything from the loaded backend into this namespace
for _name in dir(_backend):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_backend, _name)

# 4. Import your Python-side extensions
from .module import Module
from .compiler import JITCompiler
from .eager import eager_step
from .decorator import eager, jit
from .creation import zeros, ones, randn

# 5. Automatically define __all__ based on what the backend and wrappers provide
__all__ = [
    _name for _name in globals() 
    if not _name.startswith("_") and _name not in ("platform", "subprocess", "_detect_x86_microarch", "_machine", "_arch", "_backend", "_name")
]