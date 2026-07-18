"""
net/
LAN multiplayer: an authoritative server, a thin client, and the wire protocol
between them.

The shape is Minecraft's own: whoever opens their world to the network runs a
real server inside their own process (on a background thread) and plays through
a local client that talks to it exactly like a remote one does. There is no
"host mode" branch anywhere in the game logic. That is the whole point - a
second code path for the host is how you get bugs that only ever reproduce on
one machine.
"""

# What kind of session is running. Defined here rather than in main.py because
# the UI needs to ask (the pause menu offers "Open to LAN" only where opening is
# a thing that can happen), and a screen importing main.py would be a cycle.
#
# These describe the TRANSPORT, never the rules. Nothing downstream of net/ is
# allowed to branch on them to decide what happens in the world - the host and
# the guest run the same client against the same authoritative server, and the
# moment one of these turns up in an `if` inside player/ or world/, the promise
# in the paragraph above is gone. main.py reads them to decide what to
# construct, and to skip work only a server should be doing; the pause screen
# reads them to decide what to offer. That is the whole list.
MODE_SINGLEPLAYER = "singleplayer"  # own server, own loopback client, port closed
MODE_HOST = "host"                  # same, but the listen socket is open
MODE_CLIENT = "client"              # someone else's server, over a socket

