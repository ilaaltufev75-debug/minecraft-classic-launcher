"""
entity/
Things in the world that are not blocks.

Currently just the player model. The split from render/ is deliberate: what a
player is SHAPED like and how that shape MOVES is geometry and trigonometry,
and none of it needs a GL context to be correct - or to be tested. render/
player_renderer.py is the half that owns buffers and textures and cannot run
without a window.
"""
