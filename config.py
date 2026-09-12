# ===== BOT SETTINGS =====
BOT_NAME = "MyBot"
BOT_RACE = "Zerg"  # Options: Terran, Protoss, Zerg, Random

# ===== REMOTE SC2 CLIENT =====
# The SC2 game client ("ingame window") runs on another machine on the LAN.
# Launch it there listening on the network, e.g.:
#   SC2_x64.exe -listen 0.0.0.0 -port 5001
# then point this at that machine's LAN IP.
REMOTE_HOST = "192.168.1.X"  # LAN IP of the machine running SC2
REMOTE_PORT = 5001

# ===== GAME SETTINGS =====
# Map names must match .SC2Map files already present in the REMOTE machine's
# Maps folder (this machine does not need SC2 installed).
MAP_POOL = [
    "PersephoneAIE_v4",
    "PylonAIE_v4",
    "TorchesAIE_v4",
]

# ===== OPPONENT SETTINGS =====
OPPONENT_RACE = "Terran"  # Terran, Zerg, Protoss, Random
OPPONENT_DIFFICULTY = "Medium"  # VeryEasy, Easy, Medium, Hard, VeryHard, etc.

# ===== GAME MODE =====
# True to play in realtime (like a human), False for faster simulation.
REALTIME = False
