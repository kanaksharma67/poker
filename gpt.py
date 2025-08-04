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

def extract_text_regions_focused(image_path):
    """
    Extract text regions with focus on poker essentials only
    """
    print(f"\n📝 FOCUSED TEXT EXTRACTION")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Enhanced preprocessing for poker text
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    preprocessing_methods = [
        ("enhanced", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
        ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]
    
    all_text_regions = []
    seen_regions = set()
    
    for method_name, processed_img in preprocessing_methods:
        try:
            ocr_results = reader.readtext(processed_img, detail=1, paragraph=False,
                                        width_ths=0.3, height_ths=0.3)
            
            for detection in ocr_results:
                bbox_points = detection[0]
                text = detection[1].strip()
                confidence = detection[2]
                
                # Higher threshold for cleaner results
                if confidence < 0.4 or len(text) < 1:
                    continue
                
                # ONLY keep poker-relevant text
                if not is_poker_relevant_text(text):
                    continue
                
                center_x = sum([point[0] for point in bbox_points]) / 4
                center_y = sum([point[1] for point in bbox_points]) / 4
                region_key = f"{text.lower()}_{int(center_x/25)}_{int(center_y/25)}"
                
                if region_key in seen_regions:
                    continue
                seen_regions.add(region_key)
                
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
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
                
                print(f"   📝 {method_name}: '{text}' conf:{confidence:.2f}")
        except Exception as e:
            print(f"   ⚠️ Error with {method_name}: {e}")
    
    print(f"📝 Total poker-relevant regions: {len(all_text_regions)}")
    return all_text_regions

def is_poker_relevant_text(text):
    """
    Filter to keep ONLY poker-relevant text
    """
    text_lower = text.lower().strip()
    
    # Skip very long text (likely UI elements)
    if len(text) > 12:
        return False
    
    # Keep numeric values (potential bets/stacks)
    if re.search(r'\d', text):
        return True
    
    # Keep poker card values
    if is_valid_poker_card(text):
        return True
    
    # Keep action button text
    action_words = ['fold', 'call', 'raise', 'check', 'bet', 'all-in', 'allin']
    if any(word in text_lower for word in action_words):
        return True
    
    # Keep pot-related text
    if 'pot' in text_lower:
        return True
    
    # Keep game state text
    game_states = ['preflop', 'flop', 'turn', 'river']
    if any(state in text_lower for state in game_states):
        return True
    
    # Skip everything else
    return False

def detect_game_state(img):
    """
    Detect current game state (preflop, flop, turn, river)
    """
    print(f"\n🎮 DETECTING GAME STATE")
    
    height, width = img.shape[:2]
    
    # Look for game state text in top area
    top_region = img[0:int(height*0.3), :]
    gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    
    try:
        results = reader.readtext(gray, detail=1, paragraph=False)
        
        for detection in results:
            text = detection[1].strip().lower()
            confidence = detection[2]
            
            if confidence > 0.3:
                if 'preflop' in text:
                    print(f"   🎮 Game State: PREFLOP")
                    return 'preflop'
                elif 'flop' in text:
                    print(f"   🎮 Game State: FLOP")
                    return 'flop'
                elif 'turn' in text:
                    print(f"   🎮 Game State: TURN")
                    return 'turn'
                elif 'river' in text:
                    print(f"   🎮 Game State: RIVER")
                    return 'river'
    except:
        pass
    
    # Fallback: detect based on number of community cards
    community_cards = detect_community_cards_enhanced(img)
    card_count = len(community_cards)
    
    if card_count == 0:
        print(f"   🎮 Game State: PREFLOP (no community cards)")
        return 'preflop'
    elif card_count == 3:
        print(f"   🎮 Game State: FLOP (3 cards)")
        return 'flop'
    elif card_count == 4:
        print(f"   🎮 Game State: TURN (4 cards)")
        return 'turn'
    elif card_count == 5:
        print(f"   🎮 Game State: RIVER (5 cards)")
        return 'river'
    else:
        print(f"   🎮 Game State: UNKNOWN ({card_count} cards)")
        return 'unknown'

def detect_community_cards_enhanced(img):
    """
    Enhanced community card detection specifically for JungleePoker
    """
    print(f"\n🃏 ENHANCED COMMUNITY CARD DETECTION")
    
    height, width = img.shape[:2]
    
    # Multiple search regions for community cards (center focus)
    search_regions = [
        # Primary center region
        (int(height * 0.25), int(height * 0.65), int(width * 0.15), int(width * 0.85)),
        # Slightly higher center
        (int(height * 0.20), int(height * 0.60), int(width * 0.20), int(width * 0.80)),
        # Wider center region
        (int(height * 0.30), int(height * 0.70), int(width * 0.10), int(width * 0.90))
    ]
    
    all_community_cards = []
    
    for region_idx, (y1, y2, x1, x2) in enumerate(search_regions):
        print(f"   🔍 Searching region {region_idx + 1}: center area")
        
        community_region = img[y1:y2, x1:x2]
        
        # Multiple color space detections for better card finding
        cards_found = []
        
        # Method 1: Enhanced HSV detection
        cards_found.extend(detect_cards_hsv_enhanced(community_region, x1, y1))
        
        # Method 2: LAB color space detection
        cards_found.extend(detect_cards_lab_enhanced(community_region, x1, y1))
        
        # Method 3: Grayscale with multiple thresholds
        cards_found.extend(detect_cards_gray_enhanced(community_region, x1, y1))
        
        # Method 4: Edge-based detection
        cards_found.extend(detect_cards_edge_based(community_region, x1, y1))
        
        # Remove duplicates and sort by x-coordinate
        unique_cards = remove_duplicate_cards(cards_found)
        unique_cards.sort(key=lambda c: c["center_x"])
        
        print(f"   🃏 Found {len(unique_cards)} potential community cards in region {region_idx + 1}")
        
        # Process each card with enhanced OCR
        for i, card_data in enumerate(unique_cards):
            card_text = ocr_community_card_enhanced(card_data["region_image"])
            
            if card_text and is_valid_poker_card(card_text):
                card_data["text"] = card_text
                card_data["method"] = f"community_enhanced_region_{region_idx + 1}"
                card_data["card_type"] = "community"
                all_community_cards.append(card_data)
                print(f"   🃏 Community Card {len(all_community_cards)}: {card_text}")
        
        # If we found cards in this region, prefer them
        if len(all_community_cards) >= 3:
            break
    
    # Final cleanup and validation
    final_cards = []
    seen_positions = set()
    
    for card in all_community_cards:
        pos_key = f"{int(card['center_x']/30)}_{int(card['center_y']/30)}"
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            final_cards.append(card)
    
    # Sort by x-coordinate for proper order
    final_cards.sort(key=lambda c: c["center_x"])
    
    print(f"   🃏 Final community cards: {len(final_cards)}")
    return final_cards

def detect_cards_hsv_enhanced(region, offset_x, offset_y):
    """
    Enhanced HSV-based card detection
    """
    cards = []
    
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Multiple HSV ranges for different lighting conditions
    hsv_ranges = [
        # Bright white cards
        ([0, 0, 180], [180, 50, 255]),
        # Slightly dimmer cards
        ([0, 0, 160], [180, 70, 255]),
        # Very bright cards
        ([0, 0, 200], [180, 30, 255])
    ]
    
    for lower, upper in hsv_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            area = w * h
            
            # Community card criteria (slightly different from hole cards)
            if (0.55 < aspect_ratio < 0.85 and 
                1000 < area < 12000 and 
                w > 35 and h > 45):
                
                full_x = x + offset_x
                full_y = y + offset_y
                card_region = region[y:y+h, x:x+w]
                
                cards.append({
                    "bbox": {"x1": full_x, "y1": full_y, "x2": full_x + w, "y2": full_y + h},
                    "center_x": full_x + w/2,
                    "center_y": full_y + h/2,
                    "region_image": card_region,
                    "area": area,
                    "width": w,
                    "height": h,
                    "confidence": 0.8
                })
    
    return cards

def detect_cards_lab_enhanced(region, offset_x, offset_y):
    """
    Enhanced LAB color space detection
    """
    cards = []
    
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    
    # LAB ranges for white/light cards
    lab_ranges = [
        ([160, 0, 0], [255, 255, 255]),
        ([140, 0, 0], [255, 255, 255]),
        ([180, 0, 0], [255, 255, 255])
    ]
    
    for lower, upper in lab_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(lab, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            area = w * h
            
            if (0.55 < aspect_ratio < 0.85 and 
                1000 < area < 12000 and 
                w > 35 and h > 45):
                
                full_x = x + offset_x
                full_y = y + offset_y
                card_region = region[y:y+h, x:x+w]
                
                cards.append({
                    "bbox": {"x1": full_x, "y1": full_y, "x2": full_x + w, "y2": full_y + h},
                    "center_x": full_x + w/2,
                    "center_y": full_y + h/2,
                    "region_image": card_region,
                    "area": area,
                    "width": w,
                    "height": h,
                    "confidence": 0.8
                })
    
    return cards

def detect_cards_gray_enhanced(region, offset_x, offset_y):
    """
    Enhanced grayscale detection with multiple thresholds
    """
    cards = []
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Multiple threshold values
    thresholds = [170, 180, 190, 200, 210]
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            area = w * h
            
            if (0.55 < aspect_ratio < 0.85 and 
                1000 < area < 12000 and 
                w > 35 and h > 45):
                
                full_x = x + offset_x
                full_y = y + offset_y
                card_region = region[y:y+h, x:x+w]
                
                cards.append({
                    "bbox": {"x1": full_x, "y1": full_y, "x2": full_x + w, "y2": full_y + h},
                    "center_x": full_x + w/2,
                    "center_y": full_y + h/2,
                    "region_image": card_region,
                    "area": area,
                    "width": w,
                    "height": h,
                    "confidence": 0.7
                })
    
    return cards

def detect_cards_edge_based(region, offset_x, offset_y):
    """
    Edge-based card detection
    """
    cards = []
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect card boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / float(h)
        area = w * h
        
        if (0.55 < aspect_ratio < 0.85 and 
            1000 < area < 12000 and 
            w > 35 and h > 45):
            
            full_x = x + offset_x
            full_y = y + offset_y
            card_region = region[y:y+h, x:x+w]
            
            cards.append({
                "bbox": {"x1": full_x, "y1": full_y, "x2": full_x + w, "y2": full_y + h},
                "center_x": full_x + w/2,
                "center_y": full_y + h/2,
                "region_image": card_region,
                "area": area,
                "width": w,
                "height": h,
                "confidence": 0.6
            })
    
    return cards

def ocr_community_card_enhanced(card_region):
    """
    Enhanced OCR specifically for community cards with rank+suit
    """
    try:
        if card_region.shape[0] < 20 or card_region.shape[1] < 20:
            return None
        
        # Upscale for better OCR
        scale_factor = 3
        upscaled = cv2.resize(card_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Multiple preprocessing approaches
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        preprocessing_methods = [
            ("original", upscaled),
            ("gray", gray),
            ("clahe", cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4)).apply(gray)),
            ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)),
            ("inverted", cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])),
            ("gaussian", cv2.GaussianBlur(gray, (3, 3), 0)),
            ("median", cv2.medianBlur(gray, 3))
        ]
        
        all_detections = []
        
        for method_name, processed in preprocessing_methods:
            try:
                # Very sensitive OCR settings for community cards
                results = reader.readtext(processed, detail=1, paragraph=False,
                                        width_ths=0.05, height_ths=0.05,
                                        allowlist='AKQJT23456789♠♥♦♣SHDCshdc')
                
                for detection in results:
                    text = detection[1].strip().upper()
                    confidence = detection[2]
                    
                    if confidence > 0.1 and len(text) > 0:
                        all_detections.append((text, confidence, method_name))
            except:
                continue
        
        if all_detections:
            # Sort by confidence
            all_detections.sort(key=lambda x: x[1], reverse=True)
            
            # Try to find complete rank+suit combinations
            for text, conf, method in all_detections:
                formatted_card = format_community_card(text)
                if formatted_card and is_valid_poker_card(formatted_card):
                    return formatted_card
            
            # Fallback to rank-only detection
            for text, conf, method in all_detections:
                if len(text) == 1 and text in 'AKQJT23456789':
                    return text
        
        return None
    except Exception as e:
        print(f"   ⚠️ Community OCR Error: {e}")
        return None

def format_community_card(text):
    """
    Format community card text to rank+suit format
    """
    text = text.upper().strip()
    
    # Handle complete card (rank + suit)
    if len(text) >= 2:
        rank = text[0]
        suit_part = text[1:]
        
        if rank in 'AKQJT23456789':
            # Convert suit symbols/letters to standard format
            if '♠' in suit_part or 'S' in suit_part:
                return f"{rank}s"
            elif '♥' in suit_part or 'H' in suit_part:
                return f"{rank}h"
            elif '♦' in suit_part or 'D' in suit_part:
                return f"{rank}d"
            elif '♣' in suit_part or 'C' in suit_part:
                return f"{rank}c"
    
    # Handle 10 specially
    if text.startswith('10'):
        rank = 'T'
        if len(text) > 2:
            suit_part = text[2:]
            if '♠' in suit_part or 'S' in suit_part:
                return f"{rank}s"
            elif '♥' in suit_part or 'H' in suit_part:
                return f"{rank}h"
            elif '♦' in suit_part or 'D' in suit_part:
                return f"{rank}d"
            elif '♣' in suit_part or 'C' in suit_part:
                return f"{rank}c"
        return rank
    
    # Single rank
    if len(text) == 1 and text in 'AKQJT23456789':
        return text
    
    return None

def remove_duplicate_cards(cards):
    """
    Remove duplicate card detections
    """
    unique_cards = []
    
    for card in cards:
        is_duplicate = False
        
        for existing in unique_cards:
            distance = ((card["center_x"] - existing["center_x"])**2 + 
                       (card["center_y"] - existing["center_y"])**2)**0.5
            
            if distance < 40:  # Cards are close to each other
                is_duplicate = True
                # Keep the one with higher confidence or larger area
                if card.get("confidence", 0) > existing.get("confidence", 0):
                    unique_cards.remove(existing)
                    unique_cards.append(card)
                break
        
        if not is_duplicate:
            unique_cards.append(card)
    
    return unique_cards

def detect_cards_focused(image_path):
    """
    Enhanced card detection for both hole cards and community cards
    """
    print(f"\n🃏 ENHANCED CARD DETECTION (HOLE + COMMUNITY)")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    all_cards = []
    
    # Step 1: Detect community cards first (they're more important)
    community_cards = detect_community_cards_enhanced(img)
    for card in community_cards:
        card["card_category"] = "community"
    all_cards.extend(community_cards)
    
    # Step 2: Detect hole cards (bottom area)
    print(f"\n🃏 DETECTING HOLE CARDS")
    bottom_region_y = int(height * 0.55)
    hole_card_region = img[bottom_region_y:height, :]
    
    # Use existing hole card detection logic
    hole_cards = detect_hole_cards_region(hole_card_region, bottom_region_y)
    for card in hole_cards:
        card["card_category"] = "hole"
    all_cards.extend(hole_cards)
    
    print(f"   🃏 Total cards detected: {len(all_cards)} (Community: {len(community_cards)}, Hole: {len(hole_cards)})")
    return all_cards

def detect_hole_cards_region(hole_card_region, offset_y):
    """
    Detect hole cards in the bottom region
    """
    cards = []
    
    # Multiple color detection methods for hole cards
    hsv = cv2.cvtColor(hole_card_region, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 70, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # LAB color space detection
    lab = cv2.cvtColor(hole_card_region, cv2.COLOR_BGR2LAB)
    lower_light = np.array([150, 0, 0])
    upper_light = np.array([255, 255, 255])
    lab_mask = cv2.inRange(lab, lower_light, upper_light)
    
    # Grayscale threshold
    gray = cv2.cvtColor(hole_card_region, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, lab_mask)
    combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_cards = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / float(h)
        area = w * h
        
        # Hole card criteria
        if (0.4 < aspect_ratio < 1.2 and 
            800 < area < 15000 and 
            w > 25 and h > 35):
            
            full_y = y + offset_y
            card_region = hole_card_region[y:y+h, x:x+w]
            
            potential_cards.append({
                "bbox": {"x1": x, "y1": full_y, "x2": x + w, "y2": full_y + h},
                "center_x": x + w/2,
                "center_y": full_y + h/2,
                "region_image": card_region,
                "area": area,
                "width": w,
                "height": h
            })
    
    # Sort by x-coordinate
    potential_cards.sort(key=lambda c: c["center_x"])
    
    # Process each potential hole card
    for i, card_data in enumerate(potential_cards):
        card_text = ocr_card_region_enhanced(card_data["region_image"])
        
        if not card_text:
            card_text = ocr_card_region_simple(card_data["region_image"])
        
        if not card_text:
            card_text = ocr_card_region_aggressive(card_data["region_image"])
        
        if card_text:
            cards.append({
                "text": card_text,
                "bbox": card_data["bbox"],
                "center_x": card_data["center_x"],
                "center_y": card_data["center_y"],
                "confidence": 0.9,
                "region_image": card_data["region_image"],
                "method": "hole_card_detection",
                "card_position": "left" if i == 0 else "right" if i == 1 else f"card_{i+1}"
            })
            print(f"   🃏 Hole Card {i+1}: {card_text}")
    
    return cards

# Keep all the existing OCR functions unchanged
def ocr_card_region_simple(card_region):
    """
    Simple OCR approach for cards
    """
    try:
        if card_region.shape[0] < 10 or card_region.shape[1] < 10:
            return None
        
        # Simple grayscale conversion
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
        
        # Try basic OCR
        results = reader.readtext(gray, detail=1, paragraph=False)
        
        for detection in results:
            text = detection[1].strip().upper()
            confidence = detection[2]
            
            if confidence > 0.1 and is_valid_poker_card(text):
                return text
        
        return None
    except:
        return None

def ocr_card_region_aggressive(card_region):
    """
    Aggressive OCR approach for difficult cards
    """
    try:
        if card_region.shape[0] < 10 or card_region.shape[1] < 10:
            return None
        
        # Upscale the image significantly
        scale_factor = 3
        upscaled = cv2.resize(card_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # Try multiple aggressive preprocessing
        methods = [
            gray,
            cv2.GaussianBlur(gray, (3, 3), 0),
            cv2.medianBlur(gray, 3),
            cv2.bilateralFilter(gray, 9, 75, 75),
            cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8)).apply(gray),
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2),
            cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        ]
        
        all_detections = []
        
        for processed in methods:
            try:
                # Very sensitive OCR settings
                results = reader.readtext(processed, detail=1, paragraph=False,
                                         width_ths=0.05, height_ths=0.05)
                
                for detection in results:
                    text = detection[1].strip().upper()
                    confidence = detection[2]
                    
                    if confidence > 0.05 and len(text) > 0:
                        all_detections.append((text, confidence))
            except:
                continue
        
        # Process all detections to find valid cards
        if all_detections:
            # Sort by confidence
            all_detections.sort(key=lambda x: x[1], reverse=True)
            
            # Look for valid poker cards
            for text, conf in all_detections:
                if is_valid_poker_card(text):
                    return text
                
                # Try to extract card parts
                for char in text:
                    if char in 'AKQJT23456789':
                        return char
        
        return None
    except:
        return None

def detect_cards_aggressive(img):
    """
    Aggressive card detection as fallback
    """
    print("   🔍 AGGRESSIVE CARD DETECTION")
    
    height, width = img.shape[:2]
    
    # Search in multiple areas
    search_areas = [
        (int(height * 0.5), height, 0, width),  # Bottom half
        (int(height * 0.3), int(height * 0.7), 0, width),  # Middle area
        (0, int(height * 0.5), 0, width)  # Top half
    ]
    
    aggressive_cards = []
    
    for y1, y2, x1, x2 in search_areas:
        search_region = img[y1:y2, x1:x2]
        
        # Convert to different color spaces
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Very aggressive thresholding
        thresholds = [150, 160, 170, 180, 190, 200]
        
        for thresh_val in thresholds:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / float(h)
                area = w * h
                
                # Very lenient criteria
                if (0.3 < aspect_ratio < 1.5 and
                     500 < area < 20000 and
                     w > 20 and h > 25):
                    
                    # Adjust coordinates back to full image
                    full_y = y + y1
                    full_x = x + x1
                    card_region = img[full_y:full_y+h, full_x:full_x+w]
                    
                    # Try OCR
                    card_text = ocr_card_region_aggressive(card_region)
                    
                    if card_text and is_valid_poker_card(card_text):
                        # Check if we already have this card (avoid duplicates)
                        duplicate = False
                        for existing in aggressive_cards:
                            if (abs(existing["center_x"] - (full_x + w/2)) < 50 and
                                 abs(existing["center_y"] - (full_y + h/2)) < 50):
                                duplicate = True
                                break
                        
                        if not duplicate:
                            aggressive_cards.append({
                                "text": card_text,
                                "bbox": {"x1": full_x, "y1": full_y, "x2": full_x + w, "y2": full_y + h},
                                "center_x": full_x + w/2,
                                "center_y": full_y + h/2,
                                "confidence": 0.7,
                                "region_image": card_region,
                                "method": "aggressive_detection"
                            })
                            print(f"   🃏 Aggressive Card: {card_text} at ({full_x + w/2:.0f}, {full_y + h/2:.0f})")
                            
                            # Stop if we found enough cards
                            if len(aggressive_cards) >= 2:
                                return aggressive_cards
    
    return aggressive_cards

def detect_community_cards(img):
    """
    Legacy community card detection (kept for compatibility)
    """
    return detect_community_cards_enhanced(img)

def ocr_card_region_enhanced(card_region):
    """
    Enhanced OCR specifically for poker cards with suit detection
    """
    try:
        if card_region.shape[0] < 15 or card_region.shape[1] < 15:
            return None
        
        # Multiple preprocessing methods optimized for cards
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR if card is small
        if card_region.shape[0] < 60:
            scale_factor = 2
            card_region = cv2.resize(card_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        preprocessing_methods = [
            ("original", card_region),
            ("gray", gray),
            ("enhanced", cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4)).apply(gray)),
            ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)),
            ("inverted", cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
        ]
        
        detected_texts = []
        
        for method_name, processed in preprocessing_methods:
            try:
                # Use more sensitive OCR settings for cards
                results = reader.readtext(processed, detail=1, paragraph=False,
                                         width_ths=0.1, height_ths=0.1)
                
                for detection in results:
                    text = detection[1].strip().upper()
                    confidence = detection[2]
                    
                    if confidence > 0.2 and len(text) > 0:
                        detected_texts.append((text, confidence, method_name))
            except:
                continue
        
        # Process detected texts to form complete card values
        if detected_texts:
            # Sort by confidence
            detected_texts.sort(key=lambda x: x[1], reverse=True)
            
            # Try to combine rank and suit
            best_card = combine_rank_and_suit(detected_texts)
            if best_card:
                return best_card
            
            # Fallback to best single detection
            for text, conf, method in detected_texts:
                if is_valid_poker_card(text):
                    return text
        
        return None
    except Exception as e:
        print(f"   ⚠️ OCR Error: {e}")
        return None

def combine_rank_and_suit(detected_texts):
    """
    Combine detected rank and suit into complete card value
    """
    ranks = []
    suits = []
    
    for text, conf, method in detected_texts:
        text = text.upper().strip()
        
        # Check for ranks
        if text in ['A', 'K', 'Q', 'J', 'T', '10', '2', '3', '4', '5', '6', '7', '8', '9']:
            ranks.append((text, conf))
        
        # Check for suits (symbols and letters)
        if any(suit in text for suit in ['♠', '♥', '♦', '♣', 'S', 'H', 'D', 'C', 'SPADE', 'HEART', 'DIAMOND', 'CLUB']):
            suits.append((text, conf))
        
        # Check for combined rank+suit
        if len(text) >= 2:
            rank_part = text[0]
            suit_part = text[1:]
            
            if (rank_part in 'AKQJT23456789' and
                 any(s in suit_part for s in ['♠', '♥', '♦', '♣', 'S', 'H', 'D', 'C'])):
                return format_card_value(text)
    
    # Try to combine best rank with best suit
    if ranks and suits:
        best_rank = max(ranks, key=lambda x: x[1])[0]
        best_suit = max(suits, key=lambda x: x[1])[0]
        
        # Convert suit to standard format
        suit_char = convert_suit_to_char(best_suit)
        if suit_char:
            return f"{best_rank}{suit_char}"
    
    # Return best rank if no suit found
    if ranks:
        return max(ranks, key=lambda x: x[1])[0]
    
    return None

def convert_suit_to_char(suit_text):
    """
    Convert suit text to standard character
    """
    suit_text = suit_text.upper()
    
    if '♠' in suit_text or 'S' in suit_text or 'SPADE' in suit_text:
        return 's'
    elif '♥' in suit_text or 'H' in suit_text or 'HEART' in suit_text:
        return 'h'
    elif '♦' in suit_text or 'D' in suit_text or 'DIAMOND' in suit_text:
        return 'd'
    elif '♣' in suit_text or 'C' in suit_text or 'CLUB' in suit_text:
        return 'c'
    
    return None

def format_card_value(card_text):
    """
    Format card value consistently
    """
    card_text = card_text.upper().strip()
    
    # Handle 10 specially
    if card_text.startswith('10'):
        rank = 'T'
        suit_part = card_text[2:]
    else:
        rank = card_text[0]
        suit_part = card_text[1:]
    
    # Convert suit
    suit_char = convert_suit_to_char(suit_part)
    
    if suit_char:
        return f"{rank}{suit_char}"
    else:
        return rank

def is_valid_poker_card(text):
    """
    Enhanced poker card validation
    """
    if not text:
        return False
    
    text = text.upper().strip()
    
    # Single ranks
    if text in ['A', 'K', 'Q', 'J', 'T', '10', '2', '3', '4', '5', '6', '7', '8', '9']:
        return True
    
    # Rank with suit (like Ah, Ks, Td, 9c)
    if len(text) >= 2:
        rank = text[0]
        suit_part = text[1:].lower()
        
        if rank in 'AKQJT23456789':
            # Check for suit characters
            if any(s in suit_part for s in ['h', 's', 'd', 'c', '♠', '♥', '♦', '♣']):
                return True
            # Check for suit words
            if any(s in suit_part for s in ['heart', 'spade', 'diamond', 'club']):
                return True
    
    # Handle "10" with suit
    if text.startswith('10') and len(text) >= 3:
        suit_part = text[2:].lower()
        if any(s in suit_part for s in ['h', 's', 'd', 'c', '♠', '♥', '♦', '♣']):
            return True
    
    return False

def analyze_with_gpt4v_focused(region_image, text_content, position_info):
    """
    Focused GPT-4V analysis for ONLY essential poker elements
    """
    try:
        _, buffer = cv2.imencode('.jpg', region_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""Analyze this poker interface element: "{text_content}"
Position: {position_info}

Classify as ONE of these ESSENTIAL poker elements ONLY:
1. *COMMUNITY_CARD* - Playing cards at the center of the screen (3-5 cards). Example: 2♠ 3♦ 4♥ 5♣ 6♠
2. *HOLE_CARD* - Player's private cards (usually 2 cards at bottom). Example: 2♠ 3♦
3. *BET* - Betting amounts (numbers like 200, 800, 1.2K) - current bet amounts
4. *STACK* - Chip stack amounts (like 4.8K, 66.7K, 18.6K, 2.4K) - player chip stacks
5. *POT* - Pot amount (contains "Pot" like "Pot:1.1K")
6. *BUTTON* - Action buttons (FOLD, CHECK, RAISE, CALL, BET, ALL-IN)
7. *GAME_STATE* - Game phase indicators (PREFLOP, FLOP, TURN, RIVER)
8. *OTHER* - None of the above

STRICT RULES:
- *COMMUNITY_CARD*: Valid poker card in center area (A,K,Q,J,T,2-9 with optional suits)
- *HOLE_CARD*: Valid poker card in bottom area (player's cards)
- *BET*: Numeric betting amounts being wagered
- *STACK*: Chip stack amounts (usually with K suffix like 4.8K, 66.7K)
- *POT*: Must contain "Pot" word
- *BUTTON*: Action words only (fold, check, raise, call, bet, all-in)
- *GAME_STATE*: Game phase words (preflop, flop, turn, river)
- *OTHER*: Everything else (UI elements, game info, etc.)

Text: "{text_content}"
Respond with ONLY: COMMUNITY_CARD, HOLE_CARD, BET, STACK, POT, BUTTON, GAME_STATE, or OTHER"""

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

def get_position_info(center_x, center_y, width, height):
    """
    Simple position information
    """
    if center_x < width * 0.33:
        h_pos = "left"
    elif center_x < width * 0.67:
        h_pos = "center"
    else:
        h_pos = "right"
    
    if center_y < height * 0.33:
        v_pos = "top"
    elif center_y < height * 0.67:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    return f"{v_pos} {h_pos}"

def process_poker_image_focused(image_path):
    """
    Process poker image with focus on ONLY essential elements including game state
    """
    print(f"\n{'='*60}")
    print(f"🎯 FOCUSED PROCESSING: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Step 1: Detect game state
    game_state = detect_game_state(img)
    
    # Step 2: Extract focused text regions
    text_regions = extract_text_regions_focused(image_path)
    
    # Step 3: Detect cards (both hole and community)
    card_regions = detect_cards_focused(image_path)
    
    # Combine regions
    all_regions = text_regions + card_regions
    
    # Remove duplicates
    unique_regions = remove_duplicate_regions(all_regions)
    
    # Step 4: Analyze with focused GPT-4V
    results = {
        "game_state": game_state,
        "community_cards": [],
        "hole_cards": [],
        "bets": [],
        "stacks": [],
        "pot": [],
        "buttons": [],
        "other": []
    }
    
    print(f"\n🤖 ANALYZING {len(unique_regions)} REGIONS:")
    
    for region in unique_regions:
        text = region["text"]
        bbox = region["bbox"]
        center_x = region["center_x"]
        center_y = region["center_y"]
        region_image = region["region_image"]
        
        confidence = region["confidence"]
        method = region.get("method", "unknown")
        
        # Skip tiny regions
        if region_image.shape[0] < 10 or region_image.shape[1] < 10:
            continue
        
        position_info = get_position_info(center_x, center_y, width, height)
        
        # Analyze with focused GPT-4V
        element_type = analyze_with_gpt4v_focused(region_image, text, position_info)
        
        element = {
            "value": text,
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "confidence": confidence,
            "position": position_info,
            "method": method
        }
        
        # Add card category if available
        if "card_category" in region:
            element["card_category"] = region["card_category"]
        
        # Add to appropriate category
        if element_type == "COMMUNITY_CARD":
            results["community_cards"].append(element)
        elif element_type == "HOLE_CARD":
            results["hole_cards"].append(element)
        elif element_type == "BET":
            results["bets"].append(element)
        elif element_type == "STACK":
            results["stacks"].append(element)
        elif element_type == "POT":
            results["pot"].append(element)
        elif element_type == "BUTTON":
            results["buttons"].append(element)
        else:
            results["other"].append(element)
    
    # Sort community cards by x-coordinate for proper order
    results["community_cards"].sort(key=lambda x: x["center_x"])
    results["hole_cards"].sort(key=lambda x: x["center_x"])
    
    # Create focused summary
    summary = {
        "game_state": game_state,
        "total_community_cards": len(results["community_cards"]),
        "total_hole_cards": len(results["hole_cards"]),
        "total_bets": len(results["bets"]),
        "total_stacks": len(results["stacks"]),
        "total_pot": len(results["pot"]),
        "total_buttons": len(results["buttons"]),
        
        "community_card_values": [card["value"] for card in results["community_cards"]],
        "hole_card_values": [card["value"] for card in results["hole_cards"]],
        "bet_values": [bet["value"] for bet in results["bets"]],
        "stack_values": [stack["value"] for stack in results["stacks"]],
        "pot_values": [pot["value"] for pot in results["pot"]],
        "button_values": [btn["value"] for btn in results["buttons"]]
    }
    
    results["summary"] = summary
    
    print(f"\n📊 FOCUSED RESULTS:")
    print(f"   🎮 Game State: {game_state}")
    print(f"   🃏 Community Cards: {summary['total_community_cards']} - {summary['community_card_values']}")
    print(f"   🃏 Hole Cards: {summary['total_hole_cards']} - {summary['hole_card_values']}")
    print(f"   💰 Bets: {summary['total_bets']} - {summary['bet_values']}")
    print(f"   💵 Stacks: {summary['total_stacks']} - {summary['stack_values']}")
    print(f"   🏆 Pot: {summary['total_pot']} - {summary['pot_values']}")
    print(f"   🔘 Buttons: {summary['total_buttons']} - {summary['button_values']}")
    
    return results

def remove_duplicate_regions(regions):
    """
    Remove duplicate regions
    """
    unique_regions = []
    
    for region in regions:
        is_duplicate = False
        
        for existing in unique_regions:
            distance = ((region["center_x"] - existing["center_x"])**2 +
                        (region["center_y"] - existing["center_y"])**2)**0.5
            
            text_similar = region["text"].lower() == existing["text"].lower()
            
            if distance < 30 and text_similar:
                is_duplicate = True
                if region["confidence"] > existing["confidence"]:
                    unique_regions.remove(existing)
                    unique_regions.append(region)
                break
        
        if not is_duplicate:
            unique_regions.append(region)
    
    return unique_regions

def visualize_focused_detection(image_path, results):
    """
    Enhanced visualization showing community cards and hole cards separately
    """
    img = Image.open(image_path)
    width, height = img.size
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.axis('off')
    
    colors = {
        "community_cards": "#FF0000",    # Red
        "hole_cards": "#FF6600",         # Orange
        "bets": "#FFFF00",               # Yellow
        "stacks": "#FF69B4",             # Pink
        "pot": "#00FFFF",                # Cyan
        "buttons": "#00FF00"             # Green
    }
    
    def draw_elements(elements, color, prefix, font_size=12):
        for i, element in enumerate(elements):
            bbox = element["bbox"]
            value = element["value"]
            
            # Special handling for cards to show position
            if prefix in ["COMMUNITY", "HOLE"] and "card_position" in element:
                label = f"{prefix} ({element['card_position']}): {value}"
            else:
                label = f"{prefix}: {value}"
            
            rect = patches.Rectangle(
                (bbox["x1"], bbox["y1"]),
                bbox["x2"] - bbox["x1"],
                bbox["y2"] - bbox["y1"],
                linewidth=3,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect)
            
            ax.text(
                bbox["x1"], bbox["y1"] - 8,
                label,
                color=color,
                fontsize=font_size,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.8, edgecolor=color, pad=3)
            )
    
    # Draw elements
    draw_elements(results["community_cards"], colors["community_cards"], "COMMUNITY", 12)
    draw_elements(results["hole_cards"], colors["hole_cards"], "HOLE", 12)
    draw_elements(results["bets"], colors["bets"], "BET", 11)
    draw_elements(results["stacks"], colors["stacks"], "STACK", 11)
    draw_elements(results["pot"], colors["pot"], "POT", 12)
    draw_elements(results["buttons"], colors["buttons"], "BTN", 11)
    
    # Enhanced summary
    summary = results["summary"]
    
    # Show game state prominently
    ax.text(
        width/2, 20,
        f"GAME STATE: {summary['game_state'].upper()}",
        color='white',
        fontsize=18,
        weight='bold',
        ha='center',
        bbox=dict(facecolor='purple', alpha=0.9, edgecolor='white', pad=6)
    )
    
    # Show community cards prominently
    if summary['community_card_values']:
        community_text = f"COMMUNITY: {' '.join(summary['community_card_values'])}"
        ax.text(
            width/2, 50,
            community_text,
            color=colors["community_cards"],
            fontsize=16,
            weight='bold',
            ha='center',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["community_cards"], pad=6)
        )
    
    # Show hole cards
    if summary['hole_card_values']:
        hole_text = f"HOLE: {' + '.join(summary['hole_card_values'])}"
        ax.text(
            width/2, 80,
            hole_text,
            color=colors["hole_cards"],
            fontsize=16,
            weight='bold',
            ha='center',
            bbox=dict(facecolor='black', alpha=0.9, edgecolor=colors["hole_cards"], pad=6)
        )
    
    # Rest of summary
    ax.text(
        width/2, 110,
        f"BETS: {summary['total_bets']} | STACKS: {summary['total_stacks']} | POT: {summary['total_pot']} | BUTTONS: {summary['total_buttons']}",
        color='white',
        fontsize=12,
        weight='bold',
        ha='center',
        bbox=dict(facecolor='black', alpha=0.9, edgecolor='white', pad=4)
    )
    
    # Save visualization
    output_filename = os.path.basename(image_path).split('.')[0] + '_enhanced_poker_detection.png'
    output_path = os.path.join(OUTPUT_CHECK_DIR, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    print(f"✅ Enhanced visualization saved: {output_path}")

def run_focused_poker_detection():
    """
    Run enhanced poker detection system with community card focus
    """
    print("🚀 ENHANCED POKER DETECTION SYSTEM")
    print("=" * 60)
    print("🎯 ESSENTIAL ELEMENTS: COMMUNITY CARDS | HOLE CARDS | BETS | STACKS | POT | BUTTONS")
    print("🎮 GAME STATES: PREFLOP | FLOP | TURN | RIVER")
    print("🃏 COMMUNITY CARDS: 3-5 cards in center with rank+suit (3h, Ks, etc.)")
    print("🚫 IGNORES: Game IDs, UI elements, hand strength, everything else")
    print("✨ CLEAN & PRECISE: JungleePoker optimized detection")
    print("=" * 60)
    
    # Get image files
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:TEST_IMAGE_COUNT]
    
    if not image_files:
        print(f"❌ No images found in {INPUT_IMAGE_DIR}")
        return
    
    print(f"📁 Processing {len(image_files)} images with enhanced detection")
    
    # Process each image
    for filename in tqdm(image_files, desc="🔄 Processing"):
        try:
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            
            # Enhanced processing with community card focus
            results = process_poker_image_focused(image_path)
            
            # Create enhanced visualization
            visualize_focused_detection(image_path, results)
            
            # Save results
            json_filename = filename.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
            json_path = os.path.join(OUTPUT_CHECK_DIR, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Enhanced results saved: {json_path}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🎯 ENHANCED POKER DETECTION SYSTEM")
    print("🃏 Community Cards: 3-5 cards in center (3h, Ks, Td, 9c, As)")
    print("🃏 Hole Cards: 2 cards at bottom (player's private cards)")
    print("💰 Bets: Betting amounts (200, 800, etc.)")
    print("💵 Stacks: Chip stacks (4.8K, 66.7K, 18.6K, 2.4K)")
    print("🏆 Pot: Pot amounts (Pot:1.1K)")
    print("🔘 Buttons: Action buttons (FOLD, CHECK, RAISE)")
    print("🎮 Game States: PREFLOP, FLOP, TURN, RIVER")
    print("🚫 IGNORES: Everything else!")
    print("=" * 60)
    
    run_focused_poker_detection()
    
    print("\n✅ Enhanced poker detection completed!")
    print(f"📁 Check results in: {OUTPUT_CHECK_DIR}")
    print("🎯 Community cards with rank+suit detected!")
    print("🎮 Game state awareness enabled!")
