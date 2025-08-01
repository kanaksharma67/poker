import pyautogui
import time

print("Move mouse to element corners and press Ctrl+C to stop:")
try:
    while True:
        x, y = pyautogui.position()
        print(f"X: {x}, Y: {y}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Done")




CLASS_POSITIONS = {
    # Player Cards (0-1)
    0: {"name": "card_player_1", "coords": (650, 550, 700, 650)},  # Bottom-right card
    1: {"name": "card_player_2", "coords": (750, 550, 800, 650)},  # Bottom-right card
    
    # Community Cards (2-6)
    2: {"name": "card_flop_1", "coords": (400, 300, 450, 400)},   # Leftmost flop card
    3: {"name": "card_flop_2", "coords": (500, 300, 550, 400)},   # Middle flop card
    4: {"name": "card_flop_3", "coords": (600, 300, 650, 400)},   # Rightmost flop card
    5: {"name": "card_turn", "coords": (700, 300, 750, 400)},     # Turn card
    6: {"name": "card_river", "coords": (800, 300, 850, 400)},    # River card
    
    # Buttons (7-12)
    7: {"name": "fold_button", "coords": (400, 700, 500, 750)},   # Fold button
    8: {"name": "check_button", "coords": (550, 700, 650, 750)},  # Check button
    9: {"name": "call_button", "coords": (700, 700, 800, 750)},   # Call button
    10: {"name": "raise_button", "coords": (850, 700, 950, 750)}, # Raise button
    
    # Text Elements (13-16)
    13: {"name": "BB_text", "coords": (100, 100, 200, 120)},      # Big Blind text
    14: {"name": "pot_text", "coords": (500, 200, 600, 220)},     # Pot amount
    15: {"name": "game_id", "coords": (50, 50, 200, 70)},         # Game ID top-left
    16: {"name": "timer", "coords": (900, 50, 1000, 70)}          # Timer top-right
}