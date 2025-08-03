import os
import json
import base64
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import easyocr
import re

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reader = easyocr.Reader(['en'])

# Configuration
INPUT_IMAGE_DIR = "./poker_screenshots"
OUTPUT_CHECK_DIR = "./checkingimg"
os.makedirs(OUTPUT_CHECK_DIR, exist_ok=True)
TEST_IMAGE_COUNT = 3

def extract_text_regions_enhanced(image_path):
    """
    Enhanced text extraction specifically for poker interfaces
    """
    print(f"\n📝 ENHANCED TEXT EXTRACTION")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Multiple preprocessing methods for different text types
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    preprocessing_methods = [
        ("original", gray),
        ("enhanced", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
        ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("morph", cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8)))
    ]
    
    all_text_regions = []
    seen_regions = set()
    
    for method_name, processed_img in preprocessing_methods:
        try:
            # Run OCR with sensitive settings for poker text
            ocr_results = reader.readtext(processed_img, detail=1, paragraph=False,
                                        width_ths=0.3, height_ths=0.3)
            
            for detection in ocr_results:
                bbox_points = detection[0]
                text = detection[1].strip()
                confidence = detection[2]
                
                # Lower threshold for poker elements
                if confidence < 0.3 or len(text) < 1:
                    continue
                
                # Calculate center and create unique key
                center_x = sum([point[0] for point in bbox_points]) / 4
                center_y = sum([point[1] for point in bbox_points]) / 4
                region_key = f"{text.lower()}_{int(center_x/20)}_{int(center_y/20)}"
                
                if region_key in seen_regions:
                    continue
                seen_regions.add(region_key)
                
                # Calculate precise bounding box
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                # Extract region image with padding
                padding = 5
                region_y1 = max(0, y1 - padding)
                region_y2 = min(height, y2 + padding)
                region_x1 = max(0, x1 - padding)
                region_x2 = min(width, x2 + padding)
                
                region_image = img[region_y1:region_y2, region_x1:region_x2]
                
                all_text_regions.append({
                    "text": text,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": confidence,
                    "region_image": region_image,
                    "method": method_name
                })
                
                print(f"   📝 {method_name}: '{text}' conf:{confidence:.2f} at ({center_x:.0f},{center_y:.0f})")
        except Exception as e:
            print(f"   ⚠️ Error with {method_name}: {e}")
    
    print(f"📝 Total unique text regions: {len(all_text_regions)}")
    return all_text_regions

def detect_colored_regions(image_path):
    """
    Detect colored regions that might contain poker elements (like red stack boxes, yellow pot)
    """
    print(f"\n🎨 DETECTING COLORED REGIONS")
    
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    colored_regions = []
    
    # Define color ranges for poker interface elements
    color_ranges = {
        "red_stack": ([0, 100, 100], [10, 255, 255]),      # Red stack boxes
        "yellow_pot": ([20, 100, 100], [30, 255, 255]),    # Yellow pot area
        "green_button": ([40, 100, 100], [80, 255, 255]),  # Green buttons
        "orange_button": ([10, 100, 100], [20, 255, 255])  # Orange buttons
    }
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask for this color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if area > 500:  # Minimum area for poker elements
                region_image = img[y:y+h, x:x+w]
                
                # Try to OCR this colored region
                region_text = ocr_colored_region(region_image)
                
                if region_text:
                    colored_regions.append({
                        "text": region_text,
                        "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                        "center_x": x + w/2,
                        "center_y": y + h/2,
                        "confidence": 0.7,
                        "region_image": region_image,
                        "color_type": color_name,
                        "method": "color_detection"
                    })
                    print(f"   🎨 {color_name}: '{region_text}' at ({x}, {y})")
    
    return colored_regions

def ocr_colored_region(region_image):
    """
    OCR text from colored regions
    """
    try:
        if region_image.shape[0] < 10 or region_image.shape[1] < 10:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
        
        # Try different thresholding methods
        methods = [
            gray,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        for processed in methods:
            results = reader.readtext(processed, detail=1)
            for detection in results:
                text = detection[1].strip()
                confidence = detection[2]
                
                if confidence > 0.3 and len(text) > 0:
                    return text
        return None
    except:
        return None

def detect_cards_advanced(image_path):
    """
    Advanced card detection for poker interface
    """
    print(f"\n🃏 ADVANCED CARD DETECTION")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Multiple approaches for card detection
    card_masks = []
    
    # Method 1: White/light detection in HSV
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    card_masks.append(white_mask)
    
    # Method 2: Light detection in LAB
    lower_light = np.array([180, 0, 0])
    upper_light = np.array([255, 255, 255])
    light_mask = cv2.inRange(lab, lower_light, upper_light)
    card_masks.append(light_mask)
    
    # Combine masks
    combined_mask = np.zeros_like(white_mask)
    for mask in card_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_regions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check for card-like properties
        aspect_ratio = w / float(h)
        area = w * h
        
        # Playing cards criteria (adjusted for poker interface)
        if (0.5 < aspect_ratio < 0.9 and 
            1500 < area < 12000 and 
            w > 40 and h > 55):
            
            # Extract card region
            card_region = img[y:y+h, x:x+w]
            
            # OCR the card
            card_text = ocr_card_region_advanced(card_region)
            
            if card_text and is_valid_poker_card(card_text):
                card_regions.append({
                    "text": card_text,
                    "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                    "center_x": x + w/2,
                    "center_y": y + h/2,
                    "confidence": 0.8,
                    "region_image": card_region,
                    "method": "card_detection"
                })
                print(f"   🃏 Card: {card_text} at ({x}, {y}) size:{w}x{h}")
    
    return card_regions

def ocr_card_region_advanced(card_region):
    """
    Advanced OCR for card regions
    """
    try:
        # Multiple preprocessing methods
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
        
        methods = [
            ("original", card_region),
            ("gray", gray),
            ("enhanced", cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(gray)),
            ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("inverted", cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
        ]
        
        for method_name, processed in methods:
            try:
                results = reader.readtext(processed, detail=1, paragraph=False)
                for detection in results:
                    text = detection[1].strip().upper()
                    confidence = detection[2]
                    
                    if confidence > 0.25 and is_valid_poker_card(text):
                        return text
            except:
                continue
        
        return None
    except:
        return None

def is_valid_poker_card(text):
    """
    Validate poker card text
    """
    if not text:
        return False
    
    text = text.upper().strip()
    
    # Single ranks
    if text in ['A', 'K', 'Q', 'J', 'T', '10', '2', '3', '4', '5', '6', '7', '8', '9']:
        return True
    
    # Rank with suit
    if len(text) >= 2:
        rank = text[0]
        suit_part = text[1:]
        if (rank in 'AKQJT23456789' and 
            any(s in suit_part for s in ['H', 'S', 'D', 'C', '♠', '♥', '♦', '♣', 'HEARTS', 'SPADES', 'DIAMONDS', 'CLUBS'])):
            return True
    
    # Handle "10" specially
    if text.startswith('10') and len(text) >= 3:
        suit_part = text[2:]
        if any(s in suit_part for s in ['H', 'S', 'D', 'C', '♠', '♥', '♦', '♣']):
            return True
    
    return False

def analyze_with_gpt4v_enhanced(region_image, text_content, position_info):
    """
    Enhanced GPT-4V analysis with position context
    """
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', region_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""Look at this poker game interface element that contains the text: "{text_content}"

Position context: {position_info}

Analyze this element and classify it as ONE of these poker elements:

1. **BUTTON** - Action buttons like FOLD, CALL, RAISE, CHECK, BET, ALL-IN (usually green/orange colored buttons)
2. **CARD** - Playing card values like A, K, Q, J, T, 2-9 (with or without suit symbols ♠♥♦♣)
3. **BET** - Betting amounts (numbers like 100, 200, 800, 1.2K) - usually smaller amounts
4. **POT** - Pot amount (must contain "Pot" word or be in yellow pot area)
5. **PLAYER_STACK** - Player's chip stack (large amounts like 4.2K, 18.6K, usually in red boxes)
6. **VILLAIN_STACK** - Opponent's chip stack (large amounts like 66.7K, usually in red boxes)
7. **GAME_ID** - Game identification (text like "Game I'd" or long numbers in header area)
8. **OTHER** - None of the above

IMPORTANT CLASSIFICATION RULES:
- **BUTTONS**: Look for action words in colored button areas (fold, call, raise, check, bet, all-in)
- **CARDS**: Must be valid poker card ranks/suits (A,K,Q,J,T,2-9 with optional ♠♥♦♣)
- **POT**: Only if contains "Pot" word OR is in center yellow area
- **PLAYER_STACK vs VILLAIN_STACK**:
  - PLAYER_STACK: Usually bottom area or closer to action buttons
  - VILLAIN_STACK: Usually top/side areas, away from action buttons
- **BET**: Numeric amounts that are betting amounts (not stacks)
- **GAME_ID**: Game identification text, usually in header/top area

Based on the image and text "{text_content}", what poker element is this?

Respond with ONLY ONE WORD: BUTTON, CARD, BET, POT, PLAYER_STACK, VILLAIN_STACK, GAME_ID, or OTHER"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=15,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(f"      🤖 GPT-4V: '{text_content}' → {result}")
        return result
        
    except Exception as e:
        print(f"      ❌ GPT-4V Error: {e}")
        return "OTHER"

def determine_game_state(cards, buttons):
    """
    Determine poker game state based on cards and buttons
    """
    print(f"\n🎮 DETERMINING GAME STATE")
    
    # Count community cards (usually in center/upper area)
    community_cards = []
    hole_cards = []
    
    for card in cards:
        # Fix: Check if card has the required keys
        if "center_y" not in card:
            print(f"   ⚠️ Warning: Card missing center_y: {card}")
            continue
            
        center_y = card["center_y"]
        # Cards in upper half are likely community cards
        if center_y < 400:  # Adjust based on typical poker interface
            community_cards.append(card)
        else:
            hole_cards.append(card)
    
    n_community = len(community_cards)
    
    # Determine state based on community cards
    if n_community == 0:
        game_state = "preflop"
    elif n_community == 3:
        game_state = "flop"
    elif n_community == 4:
        game_state = "turn"
    elif n_community == 5:
        game_state = "river"
    else:
        game_state = "unknown"
    
    # Check if it's player's turn based on active buttons
    player_turn = False
    active_buttons = []
    
    for button in buttons:
        # Fix: Check if button has the required keys
        if "value" not in button:
            print(f"   ⚠️ Warning: Button missing value: {button}")
            continue
            
        button_text = button["value"].lower()
        if button_text in ['fold', 'call', 'raise', 'check', 'bet', 'all-in']:
            active_buttons.append(button_text)
            player_turn = True
    
    print(f"   🎮 Community cards: {len(community_cards)}")
    print(f"   🎮 Hole cards: {len(hole_cards)}")
    print(f"   🎮 Game state: {game_state}")
    print(f"   🎮 Player turn: {player_turn}")
    print(f"   🎮 Active buttons: {active_buttons}")
    
    return {
        "state": game_state,
        "community_cards": len(community_cards),
        "hole_cards": len(hole_cards),
        "player_turn": player_turn,
        "active_buttons": active_buttons
    }

def get_position_info(center_x, center_y, width, height):
    """
    Get detailed position information for GPT-4V context
    """
    # Horizontal position
    if center_x < width * 0.25:
        h_pos = "far left"
    elif center_x < width * 0.4:
        h_pos = "left"
    elif center_x < width * 0.6:
        h_pos = "center"
    elif center_x < width * 0.75:
        h_pos = "right"
    else:
        h_pos = "far right"
    
    # Vertical position
    if center_y < height * 0.2:
        v_pos = "top header"
    elif center_y < height * 0.4:
        v_pos = "upper area"
    elif center_y < height * 0.6:
        v_pos = "center area"
    elif center_y < height * 0.8:
        v_pos = "lower area"
    else:
        v_pos = "bottom area"
    
    return f"{v_pos}, {h_pos} of screen"

def process_poker_image_advanced(image_path):
    """
    Advanced poker image processing
    """
    print(f"\n{'='*70}")
    print(f"🎯 ADVANCED PROCESSING: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Step 1: Extract text regions
    text_regions = extract_text_regions_enhanced(image_path)
    
    # Step 2: Detect colored regions
    colored_regions = detect_colored_regions(image_path)
    
    # Step 3: Detect cards
    card_regions = detect_cards_advanced(image_path)
    
    # Combine all regions
    all_regions = text_regions + colored_regions + card_regions
    
    # Remove duplicates based on proximity
    unique_regions = remove_duplicate_regions(all_regions)
    
    # Step 4: Analyze each region with GPT-4V
    results = {
        "buttons": [],
        "cards": [],
        "bets": [],
        "pot": [],
        "player_stack": [],
        "villain_stack": [],
        "game_id": [],
        "other": []
    }
    
    print(f"\n🤖 ANALYZING {len(unique_regions)} REGIONS WITH GPT-4V:")
    
    for region in unique_regions:
        text = region["text"]
        bbox = region["bbox"]
        center_x = region["center_x"]
        center_y = region["center_y"]
        region_image = region["region_image"]
        confidence = region["confidence"]
        method = region.get("method", "unknown")
        
        # Skip tiny regions
        if region_image.shape[0] < 8 or region_image.shape[1] < 8:
            continue
        
        # Get position context
        position_info = get_position_info(center_x, center_y, width, height)
        
        # Analyze with GPT-4V
        element_type = analyze_with_gpt4v_enhanced(region_image, text, position_info)
        
        # Create element object with consistent structure
        element = {
            "value": text,
            "bbox": bbox,
            "center_x": center_x,  # Fix: Ensure center_x is included
            "center_y": center_y,  # Fix: Ensure center_y is included
            "confidence": confidence,
            "position": position_info,
            "method": method
        }
        
        # Add to appropriate category
        if element_type == "BUTTON":
            results["buttons"].append(element)
        elif element_type == "CARD":
            results["cards"].append(element)
        elif element_type == "BET":
            results["bets"].append(element)
        elif element_type == "POT":
            results["pot"].append(element)
        elif element_type == "PLAYER_STACK":
            results["player_stack"].append(element)
        elif element_type == "VILLAIN_STACK":
            results["villain_stack"].append(element)
        elif element_type == "GAME_ID":
            results["game_id"].append(element)
        else:
            results["other"].append(element)
    
    # Step 5: Determine game state
    game_state_info = determine_game_state(results["cards"], results["buttons"])
    results["game_state"] = game_state_info
    
    # Step 6: Create comprehensive summary
    summary = {
        "total_buttons": len(results["buttons"]),
        "total_cards": len(results["cards"]),
        "total_bets": len(results["bets"]),
        "total_pot": len(results["pot"]),
        "total_player_stack": len(results["player_stack"]),
        "total_villain_stack": len(results["villain_stack"]),
        "total_game_id": len(results["game_id"]),
        
        "button_values": [btn["value"] for btn in results["buttons"]],
        "card_values": [card["value"] for card in results["cards"]],
        "bet_values": [bet["value"] for bet in results["bets"]],
        "pot_values": [pot["value"] for pot in results["pot"]],
        "player_stack_values": [stack["value"] for stack in results["player_stack"]],
        "villain_stack_values": [stack["value"] for stack in results["villain_stack"]],
        "game_id_values": [gid["value"] for gid in results["game_id"]],
        
        "game_state": game_state_info["state"],
        "player_turn": game_state_info["player_turn"],
        "active_buttons": game_state_info["active_buttons"]
    }
    
    results["summary"] = summary
    
    print(f"\n📊 FINAL ADVANCED RESULTS:")
    print(f"   🔘 Buttons: {summary['total_buttons']} - {summary['button_values']}")
    print(f"   🃏 Cards: {summary['total_cards']} - {summary['card_values']}")
    print(f"   💰 Bets: {summary['total_bets']} - {summary['bet_values']}")
    print(f"   🏆 Pot: {summary['total_pot']} - {summary['pot_values']}")
    print(f"   💵 Player Stack: {summary['total_player_stack']} - {summary['player_stack_values']}")
    print(f"   💵 Villain Stack: {summary['total_villain_stack']} - {summary['villain_stack_values']}")
    print(f"   🆔 Game ID: {summary['total_game_id']} - {summary['game_id_values']}")
    print(f"   🎮 Game State: {summary['game_state']}")
    print(f"   🎮 Player Turn: {summary['player_turn']}")
    print(f"   🎮 Active Buttons: {summary['active_buttons']}")
    
    return results

def remove_duplicate_regions(regions):
    """
    Remove duplicate regions based on proximity and text similarity
    """
    unique_regions = []
    
    for region in regions:
        is_duplicate = False
        
        for existing in unique_regions:
            # Check if regions are close and have similar text
            distance = ((region["center_x"] - existing["center_x"])**2 + 
                       (region["center_y"] - existing["center_y"])**2)**0.5
            
            text_similar = region["text"].lower() == existing["text"].lower()
            
            if distance < 30 and text_similar:
                is_duplicate = True
                # Keep the one with higher confidence
                if region["confidence"] > existing["confidence"]:
                    unique_regions.remove(existing)
                    unique_regions.append(region)
                break
        
        if not is_duplicate:
            unique_regions.append(region)
    
    return unique_regions

def visualize_advanced_detection(image_path, results):
    """
    Advanced visualization with game state
    """
    img = Image.open(image_path)
    width, height = img.size
    
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.imshow(img)
    ax.axis('off')
    
    # Enhanced color scheme
    colors = {
        "buttons": "#00FF00",        # Bright Green
        "cards": "#FF0000",          # Red
        "bets": "#FFFF00",           # Yellow
        "pot": "#00FFFF",            # Cyan
        "player_stack": "#FF69B4",   # Hot Pink
        "villain_stack": "#FFA500",  # Orange
        "game_id": "#9370DB"         # Purple
    }
    
    def draw_elements_advanced(elements, color, prefix, font_size=11):
        for element in elements:
            bbox = element["bbox"]
            value = element["value"]
            position = element.get("position", "")
            method = element.get("method", "")
            
            # Draw rectangle with thick border
            rect = patches.Rectangle(
                (bbox["x1"], bbox["y1"]), 
                bbox["x2"] - bbox["x1"], 
                bbox["y2"] - bbox["y1"],
                linewidth=3, 
                edgecolor=color, 
                facecolor="none"
            )
            ax.add_patch(rect)
            
            # Enhanced label
            label = f"{prefix}: {value}"
            if method == "color_detection":
                label += " (COLOR)"
            elif method == "card_detection":
                label += " (CV)"
            
            ax.text(
                bbox["x1"], bbox["y1"] - 10,
                label,
                color=color,
                fontsize=font_size,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.85, edgecolor=color, pad=4)
            )
    
    # Draw all elements
    draw_elements_advanced(results["buttons"], colors["buttons"], "BUTTON", 12)
    draw_elements_advanced(results["cards"], colors["cards"], "CARD", 12)
    draw_elements_advanced(results["bets"], colors["bets"], "BET", 11)
    draw_elements_advanced(results["pot"], colors["pot"], "POT", 13)
    draw_elements_advanced(results["player_stack"], colors["player_stack"], "PLAYER", 11)
    draw_elements_advanced(results["villain_stack"], colors["villain_stack"], "VILLAIN", 11)
    draw_elements_advanced(results["game_id"], colors["game_id"], "GAME_ID", 10)
    
    # Enhanced summary display
    summary = results["summary"]
    game_state = results["game_state"]
    
    # Game state - prominent display
    state_color = "#FF0000" if game_state["player_turn"] else "#808080"
    ax.text(
        width/2, 50,
        f"GAME STATE: {summary['game_state'].upper()} | PLAYER TURN: {game_state['player_turn']}",
        color='white',
        fontsize=18,
        weight='bold',
        ha='center',
        bbox=dict(facecolor=state_color, alpha=0.95, edgecolor='white', pad=8)
    )
    
    # Active buttons
    if game_state["active_buttons"]:
        ax.text(
            width/2, 90,
            f"ACTIVE: {', '.join(game_state['active_buttons']).upper()}",
            color=colors["buttons"],
            fontsize=14,
            weight='bold',
            ha='center',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["buttons"], pad=5)
        )
    
    # Element counts
    counts_text = f"Buttons:{summary['total_buttons']} | Cards:{summary['total_cards']} | Bets:{summary['total_bets']} | Pot:{summary['total_pot']} | Player:{summary['total_player_stack']} | Villain:{summary['total_villain_stack']} | ID:{summary['total_game_id']}"
    ax.text(
        10, height - 60,
        counts_text,
        color='white',
        fontsize=12,
        weight='bold',
        bbox=dict(facecolor='black', alpha=0.9, edgecolor='white', pad=4)
    )
    
    # Values display
    if summary['pot_values']:
        ax.text(
            width/2, 130,
            f"POT: {', '.join(summary['pot_values'])}",
            color=colors["pot"],
            fontsize=16,
            weight='bold',
            ha='center',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["pot"], pad=5)
        )
    
    # Stack information
    stack_y = height - 30
    if summary['player_stack_values']:
        ax.text(
            width * 0.2, stack_y,
            f"PLAYER: {', '.join(summary['player_stack_values'])}",
            color=colors["player_stack"],
            fontsize=12,
            weight='bold',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["player_stack"], pad=3)
        )
    
    if summary['villain_stack_values']:
        ax.text(
            width * 0.8, stack_y,
            f"VILLAIN: {', '.join(summary['villain_stack_values'])}",
            color=colors["villain_stack"],
            fontsize=12,
            weight='bold',
            ha='right',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["villain_stack"], pad=3)
        )
    
    # Save visualization
    output_filename = os.path.basename(image_path).split('.')[0] + '_advanced_poker_detection.png'
    output_path = os.path.join(OUTPUT_CHECK_DIR, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    
    print(f"✅ Advanced visualization saved: {output_path}")

def run_advanced_poker_detection():
    """
    Run advanced poker detection system
    """
    print("🚀 ADVANCED POKER DETECTION SYSTEM")
    print("=" * 70)
    print("📝 Enhanced text extraction with multiple methods")
    print("🎨 Colored region detection for poker interface elements")
    print("🃏 Advanced card detection with multiple color spaces")
    print("🤖 GPT-4V analysis with position context")
    print("🎮 Game state determination")
    print("🎯 Target: BUTTONS | CARDS | BET | POT | PLAYER_STACK | VILLAIN_STACK | GAME_ID | GAME_STATE")
    print("=" * 70)
    
    # Get image files
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:TEST_IMAGE_COUNT]
    
    if not image_files:
        print(f"❌ No images found in {INPUT_IMAGE_DIR}")
        return
    
    print(f"📁 Processing {len(image_files)} images with advanced detection")
    
    # Process each image
    for filename in tqdm(image_files, desc="🔄 Processing"):
        try:
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            
            # Advanced processing
            results = process_poker_image_advanced(image_path)
            
            # Create advanced visualization
            visualize_advanced_detection(image_path, results)
            
            # Save comprehensive results
            json_filename = filename.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
            json_path = os.path.join(OUTPUT_CHECK_DIR, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Advanced results saved: {json_path}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🎯 ADVANCED POKER DETECTION SYSTEM")
    print("🎨 Detects colored regions (red stacks, yellow pot, green buttons)")
    print("🃏 Advanced card detection with multiple methods")
    print("🤖 GPT-4V with position context for accurate classification")
    print("🎮 Game state detection (preflop/flop/turn/river + player turn)")
    print("📊 Comprehensive analysis matching your poker interface!")
    print("=" * 70)
    
    run_advanced_poker_detection()
    
    print("\n✅ Advanced poker detection completed!")
    print(f"📁 Check comprehensive results in: {OUTPUT_CHECK_DIR}")
    print("🎯 Should detect exactly like your poker interface!")
