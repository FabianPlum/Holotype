"""Helpers shared by the two AntScan commands (fitter_3d.antscan_joints.run, fitter_3d.antscan_registration.run): the
names file and the lock of an output folder. Standard library only; importing this module changes nothing."""

import json
import os
import socket
import time


class UsageError(Exception):
    """wrong arguments or a folder locked by another run (exit code 2 of both commands)"""


def check_name(name):
    """A specimen name is a file key: it is joined into output paths that are later removed, so a name that is a path
    ('a/b', '..', an absolute path) is refused."""
    seps = [s for s in (os.sep, os.altsep, "/", "\\") if s]
    if name in (".", "..") or any(s in name for s in seps) or "\0" in name or os.path.isabs(name):
        raise UsageError(f"specimen name {name!r} is not a plain file key (no path separators, not '.' or '..')")
    return name


def read_names(path):
    """one specimen name per line; blank lines and lines starting with '#' are skipped; a name that occurs twice is
    taken once. A name is a file key only (check_name)."""
    out = []
    with open(path) as fh:
        for ln in fh:
            n = ln.strip()
            if n and not n.startswith("#") and n not in out:
                out.append(check_name(n))
    return out


def pid_alive(pid):
    if os.name != "posix":
        return True  # cannot be told without side effects: treat the lock as held (--break-lock takes it over)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


def acquire_lock(out, break_lock=False):
    """OUT_DIR/.lock: one run at a time writes to an output folder. The lock file is created atomically (O_EXCL). A
    lock whose process is gone (same host) is taken over; any other lock stops the run unless break_lock is given.
    Returns the lock path (release it with release_lock)."""
    path = os.path.join(out, ".lock")
    me = dict(pid=os.getpid(), host=socket.gethostname(), started=time.strftime("%F %T"))
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(path) as fh:
                    other = json.load(fh)
            except (OSError, ValueError):
                other = {}
            if not isinstance(other, dict):
                other = {}
            stale = other.get("host") == me["host"] and not pid_alive(other.get("pid"))
            if not (stale or break_lock):
                raise UsageError(
                    f"{out} is locked by another run (pid {other.get('pid')} on {other.get('host')}, started "
                    f"{other.get('started')}); two runs into one folder would mix their outputs. Wait for it, use "
                    f"another --out, or pass --break-lock if that run is gone"
                ) from None
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(me, f)
        return path
    raise UsageError(f"could not take the lock {path}")


def release_lock(path):
    """remove the lock file, but only when this process holds it"""
    try:
        with open(path) as fh:
            mine = json.load(fh).get("pid") == os.getpid()
        if mine:
            os.remove(path)
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def json_default(x):
    """`default=` of json.dump for the AntScan outputs: numpy scalars and arrays as Python values, anything else as
    its str()"""
    import numpy as np

    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)
