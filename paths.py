"""
paths.py
Where things are, answered once, so nothing else has to guess.

THE BUG THIS EXISTS TO KILL
---------------------------
Every asset in this project used to be reached by a relative path:

    BLOCK_TEXTURES_DIR = os.path.join("assets", "textures", "blocks")

A relative path is resolved against the CURRENT WORKING DIRECTORY, which is
whatever folder the process happened to be launched from - not where the code
lives. Run from PyCharm, the working directory is the project root and this
works. Hand the game to someone as an .exe and it does not, because:

  - PyInstaller unpacks the bundled files into a temporary folder and points
    sys._MEIPASS at it. The exe's working directory is wherever the person
    double-clicked it from - their Desktop, usually. So `assets/textures/blocks`
    resolves to `C:/Users/Them/Desktop/assets/textures/blocks`, which does not
    exist, and every disk-backed texture silently falls back to the procedural
    one. Nothing crashes. Nothing is logged. The textures are just gone.

This is why adding `datas` to the .spec on its own does not fix it: putting the
files in the bundle is necessary, but the code still has to look where they were
put.

TWO ROOTS, NOT ONE
------------------
They are different directories, and conflating them is the second half of the
same bug:

  asset_path()  - things shipped WITH the game and never written to. Inside the
                  bundle (sys._MEIPASS) when frozen.
  data_path()   - things the game writes: saves, settings. These must NOT go in
                  the bundle. In a onefile build sys._MEIPASS is a temp folder
                  that is deleted on exit, so a world saved there is gone the
                  moment the player quits - and they would only find that out
                  after building something for an hour.

data_path lands next to the .exe, which is what a game you pass to a friend as
a zip should do: everything stays together, and deleting the folder deletes the
game. The trade-off is that it needs a writable location - fine for a Desktop or
a games folder, not for Program Files. If that ever bites, this is the one
function to change.
"""

import os
import sys


def is_frozen() -> bool:
    """True inside a PyInstaller build, False when run from source."""
    return getattr(sys, "frozen", False)


def _bundle_root() -> str:
    if is_frozen():
        # onefile: the temp extraction dir. onedir: the _internal folder.
        # PyInstaller sets _MEIPASS for both, so this needs no branch - the
        # fallback is only for a frozen build made by something that is not
        # PyInstaller.
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    # Not CWD: this file's own folder. That makes running the game from any
    # working directory work, which the relative paths never did either - it was
    # just never noticed, because PyCharm always launches from the project root.
    return os.path.dirname(os.path.abspath(__file__))


def _writable_root() -> str:
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def asset_path(*parts) -> str:
    """A read-only file shipped with the game, e.g.
    asset_path("assets", "textures", "blocks", "grass", "up.png")."""
    return os.path.join(_bundle_root(), *parts)


def data_path(*parts) -> str:
    """A file the game writes. Creates the parent directory: every caller
    wanted that anyway, and the ones that forgot were a crash on first run in a
    fresh install."""
    path = os.path.join(_writable_root(), *parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def data_dir(*parts) -> str:
    """Like data_path, but the path itself is the directory to create."""
    path = os.path.join(_writable_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path
