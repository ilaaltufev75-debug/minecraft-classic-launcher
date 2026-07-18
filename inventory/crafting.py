"""
inventory/crafting.py
Two crafting surfaces:
  - A 2x2 personal grid, always available from the inventory screen (no
    table required).
  - A 3x3 grid, available only when interacting with a placed
    Block.CRAFTING_TABLE, which unlocks the full recipe list.

Shaped recipes match like real Minecraft: the pattern can sit ANYWHERE
within the grid (not pinned to the top-left corner) and can be mirrored
left-right (so an axe/pickaxe/etc built as a mirror image of the reference
pattern still matches) - only the RELATIVE arrangement of ingredients
matters, never which exact cells they're in. This is done by trimming both
the player's grid and the recipe pattern down to their minimal bounding
box before comparing, and trying the mirrored version too.
"""

from world.blocks import Block as B, Item as I

_EMPTY = None

RECIPES_2X2 = [
    {"shapeless": [B.WOOD_LOG], "output": {"id": B.PLANKS, "count": 4}},

    {"pattern": [[B.PLANKS], [B.PLANKS]],
     "output": {"id": I.STICK, "count": 4}},

    {"pattern": [[B.PLANKS, B.PLANKS], [B.PLANKS, B.PLANKS]],
     "output": {"id": B.CRAFTING_TABLE, "count": 1}},
]

RECIPES_3X3 = [
    {"shapeless": [B.WOOD_LOG], "output": {"id": B.PLANKS, "count": 4}},
    {"pattern": [[B.PLANKS], [B.PLANKS]],
     "output": {"id": I.STICK, "count": 4}},
    {"pattern": [[B.PLANKS, B.PLANKS], [B.PLANKS, B.PLANKS]],
     "output": {"id": B.CRAFTING_TABLE, "count": 1}},

    {"pattern": [[B.PLANKS, B.PLANKS, B.PLANKS],
                 [_EMPTY, I.STICK, _EMPTY],
                 [_EMPTY, I.STICK, _EMPTY]],
     "output": {"id": I.WOODEN_PICKAXE, "count": 1}},
    {"pattern": [[B.COBBLESTONE, B.COBBLESTONE, B.COBBLESTONE],
                 [_EMPTY, I.STICK, _EMPTY],
                 [_EMPTY, I.STICK, _EMPTY]],
     "output": {"id": I.STONE_PICKAXE, "count": 1}},

    {"pattern": [[B.PLANKS, B.PLANKS],
                 [B.PLANKS, I.STICK],
                 [_EMPTY, I.STICK]],
     "output": {"id": I.WOODEN_AXE, "count": 1}},
    {"pattern": [[B.COBBLESTONE, B.COBBLESTONE],
                 [B.COBBLESTONE, I.STICK],
                 [_EMPTY, I.STICK]],
     "output": {"id": I.STONE_AXE, "count": 1}},

    {"pattern": [[B.PLANKS],
                 [I.STICK],
                 [I.STICK]],
     "output": {"id": I.WOODEN_SHOVEL, "count": 1}},
    {"pattern": [[B.COBBLESTONE],
                 [I.STICK],
                 [I.STICK]],
     "output": {"id": I.STONE_SHOVEL, "count": 1}},

    {"pattern": [[B.PLANKS],
                 [B.PLANKS],
                 [I.STICK]],
     "output": {"id": I.WOODEN_SWORD, "count": 1}},
    {"pattern": [[B.COBBLESTONE],
                 [B.COBBLESTONE],
                 [I.STICK]],
     "output": {"id": I.STONE_SWORD, "count": 1}},

    {"pattern": [[B.PLANKS, B.PLANKS],
                 [B.PLANKS, B.PLANKS],
                 [B.PLANKS, B.PLANKS]],
     "output": {"id": B.DOOR, "count": 3}},

    {"pattern": [[B.PLANKS, I.STICK, B.PLANKS],
                 [B.PLANKS, I.STICK, B.PLANKS]],
     "output": {"id": B.FENCE, "count": 3}},

    {"pattern": [[B.PLANKS, _EMPTY, _EMPTY],
                 [B.PLANKS, B.PLANKS, _EMPTY],
                 [B.PLANKS, B.PLANKS, B.PLANKS]],
     "output": {"id": B.STAIRS_WOOD, "count": 4}},
    {"pattern": [[B.COBBLESTONE, _EMPTY, _EMPTY],
                 [B.COBBLESTONE, B.COBBLESTONE, _EMPTY],
                 [B.COBBLESTONE, B.COBBLESTONE, B.COBBLESTONE]],
     "output": {"id": B.STAIRS_STONE, "count": 4}},
]


def _bounding_box(grid):
    """grid: list of rows (lists), possibly containing None. Returns the
    trimmed sub-grid covering only rows/cols that contain at least one
    non-None entry, or None if the grid is entirely empty."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is not None:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return None
    return [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]


def _mirror(grid):
    return [list(reversed(row)) for row in grid]


def _grids_equal(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for va, vb in zip(ra, rb):
            if (va or None) != (vb or None):
                return False
    return True


def _grid_to_pattern(grid, size):
    return [grid[r * size:(r + 1) * size] for r in range(size)]


def _match(grid_stacks, recipes, size):
    """
    grid_stacks: list of size*size slots, each {"id","count"}|None, row-major.
    Returns the matching recipe's output dict, or None. Shaped recipes match
    anywhere in the grid (translation-invariant) and mirrored left-right.
    """
    ids = [s["id"] if s else None for s in grid_stacks]
    full_grid = _grid_to_pattern(ids, size)
    trimmed = _bounding_box(full_grid)

    for recipe in recipes:
        if "pattern" in recipe:
            if trimmed is None:
                continue
            recipe_trimmed = _bounding_box(recipe["pattern"])
            if recipe_trimmed is None:
                continue
            if _grids_equal(trimmed, recipe_trimmed) or _grids_equal(trimmed, _mirror(recipe_trimmed)):
                return recipe["output"]
        elif "shapeless" in recipe:
            present = [i for i in ids if i is not None]
            if len(present) != len(recipe["shapeless"]):
                continue
            if sorted(present) == sorted(recipe["shapeless"]):
                return recipe["output"]
    return None


def match_recipe(grid_stacks):
    """2x2 personal-grid match."""
    return _match(grid_stacks, RECIPES_2X2, 2)


def match_recipe_3x3(grid_stacks):
    """3x3 crafting-table match."""
    return _match(grid_stacks, RECIPES_3X3, 3)


RECIPES = RECIPES_2X2

