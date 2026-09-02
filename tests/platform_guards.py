"""Host capability probes for platform-dependent security tests.

The filesystem security tests exercise symlink rejection and
descriptor-anchored replacement semantics. Creating symlinks on Windows
requires developer mode or administrator privileges, and files kept open by
another handle cannot be replaced or deleted there, so those specific
assertions are skipped per-host instead of failing. POSIX hosts run them
unchanged.
"""

import sys
import tempfile
from pathlib import Path


def _probe_symlinks() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "target"
        target.touch()
        link = Path(directory) / "link"
        try:
            link.symlink_to(target)
        except OSError:
            return False
        return link.is_symlink()


SYMLINKS_SUPPORTED = _probe_symlinks()

#: Replacing or deleting a file that another handle keeps open, and staying
#: anchored to a directory whose path was replaced, are POSIX fd semantics.
FD_ANCHORED_SEMANTICS = sys.platform != "win32"
