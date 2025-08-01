import cv2
img = cv2.imread("test.png")

CLASS_POSITIONS = {
    # Player Cards (0-1) - Bottom center
    0: {"name": "card_player_1", "coords": (550, 500, 620, 600)},
    1: {"name": "card_player_2", "coords": (650, 500, 720, 600)},
    
    # Community Cards (2-6) - Center table
    2: {"name": "card_flop_1", "coords": (350, 250, 420, 350)},
    3: {"name": "card_flop_2", "coords": (450, 250, 520, 350)},
    4: {"name": "card_flop_3", "coords": (550, 250, 620, 350)},
    5: {"name": "card_turn", "coords": (650, 250, 720, 350)},
    6: {"name": "card_river", "coords": (750, 250, 820, 350)},
    
    # Action Buttons (7-12) - Bottom action bar
    7: {"name": "fold_button", "coords": (300, 650, 400, 700)},
    8: {"name": "check_button", "coords": (450, 650, 550, 700)},
    9: {"name": "call_button", "coords": (600, 650, 700, 700)},
    10: {"name": "raise_button", "coords": (750, 650, 850, 700)},
    
    # Text Elements (13-16)
    13: {"name": "BB_text", "coords": (150, 150, 250, 180)},       # "100/200" text
    14: {"name": "pot_text", "coords": (500, 200, 600, 230)},      # "Pot: 700" 
    15: {"name": "game_id", "coords": (50, 30, 200, 60)},          # Game ID top-left
    16: {"name": "timer", "coords": (850, 30, 950, 60)},           # "11:18 AM" top-right
    
    # Additional elements from your screenshot
    17: {"name": "player_stack_1", "coords": (200, 400, 300, 430)},  # "19.9K" stack
    18: {"name": "player_stack_2", "coords": (800, 400, 900, 430)},  # "25K" stack
    19: {"name": "straddle_button", "coords": (400, 550, 500, 580)}  # Straddle option
}
for class_id, data in CLASS_POSITIONS.items():
    x1, y1, x2, y2 = data["coords"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, data["name"], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("debug.jpg", img)