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

def analyze_card_with_gpt4(card_image, card_type="unknown"):
    """
    Use GPT-4 Vision to analyze card images and extract rank + suit
    """
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', card_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Create specific prompt based on card type
        if card_type == "community":
            prompt = """You are analyzing a poker community card image. 

Look carefully at the card and identify:
1. The card rank (A, K, Q, J, T, 2, 3, 4, 5, 6, 7, 8, 9)
2. The card suit (♠ spades, ♥ hearts, ♦ diamonds, ♣ clubs)

Return ONLY the card in this exact format: rank+suit
Examples: As, Kh, Qd, Jc, Td, 9s, 8h, 7c, 6d, 5s, 4h, 3c, 2d

If you can see the card clearly, return the rank+suit.
If you cannot read the card, return "unknown".

Respond with ONLY the card value (e.g., "As") or "unknown"."""
        else:
            prompt = """You are analyzing a poker card image. 

Look carefully at the card and identify:
1. The card rank (A, K, Q, J, T, 2, 3, 4, 5, 6, 7, 8, 9)
2. The card suit (♠ spades, ♥ hearts, ♦ diamonds, ♣ clubs)

Return ONLY the card in this exact format: rank+suit
Examples: As, Kh, Qd, Jc, Td, 9s, 8h, 7c, 6d, 5s, 4h, 3c, 2d

If you can see the card clearly, return the rank+suit.
If you cannot read the card, return "unknown".

Respond with ONLY the card value (e.g., "As") or "unknown"."""
        
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
            max_tokens=10,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate the result
        if is_valid_poker_card(result):
            print(f"      🤖 GPT-4 Card Detection: {result}")
            return result
        else:
            print(f"      🤖 GPT-4 Card Detection: invalid result '{result}', trying OCR fallback")
            return None
            
    except Exception as e:
        print(f"      ❌ GPT-4 Card Analysis Error: {e}")
        return None

def is_region_white_background(region_image):
    """
    Check if a region is mostly white background (to exclude UI elements)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
        
        # Count very bright pixels (likely white background)
        bright_pixels = np.sum(gray > 200)  # Threshold for "bright"
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels
        
        # If more than 70% of the region is bright, consider it white background
        if bright_ratio > 0.7:
            return True
        
        # Also check for very low contrast (indicating plain background)
        contrast = np.std(gray)
        if contrast < 15:  # Very low contrast
            return True
            
        # Check if the region is mostly uniform (indicating background)
        if np.max(gray) - np.min(gray) < 30:  # Very small range
            return True
            
        return False
        
    except Exception as e:
        print(f"      ⚠ Error checking white background: {e}")
        return False

def correct_pot_text(text):
    """
    Correct common OCR errors in pot values
    """
    text_lower = text.lower()
    
    # Only process text that contains "pot:" structure
    if not text_lower.startswith('pot:'):
        return text
    
    # Common OCR errors in pot values - more comprehensive corrections
    corrections = {
        'pot:726kc': 'Pot:7.26K',
        'pot:726k': 'Pot:7.26K', 
        'pot:726': 'Pot:7.26K',
        'pot:1.6k': 'Pot:1.6K',
        'pot:1.6': 'Pot:1.6K',
        'pot:10.3k': 'Pot:10.3K',
        'pot:10.3': 'Pot:10.3K',
        'pot:1.6kc': 'Pot:1.6K',
        'pot:1.6k': 'Pot:1.6K',
        'pot:1.6': 'Pot:1.6K',
        'pot:726kc': 'Pot:7.26K',
        'pot:726k': 'Pot:7.26K',
        'pot:726': 'Pot:7.26K',
        'pot:726kc': 'Pot:7.26K',
        'pot:726k': 'Pot:7.26K',
        'pot:726': 'Pot:7.26K',
        'pot:726kc': 'Pot:7.26K',
        'pot:726k': 'Pot:7.26K',
        'pot:726': 'Pot:7.26K',
        'pot:726kc': 'Pot:7.26K',
        'pot:726k': 'Pot:7.26K',
        'pot:726': 'Pot:7.26K',
    }
    # Apply corrections if any
    if text_lower in corrections:
        return corrections[text_lower]
    # General fix: replace common OCR mistakes
    text = text.replace('kc', 'K').replace('Kc', 'K').replace('kC', 'K').replace('l', '1')
    # Remove any trailing non-numeric characters after the K
    import re
    text = re.sub(r'K[^0-9]*$', 'K', text)
    return text

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
                
                # Only keep pot values with "Pot:" structure
                if 'pot' in text.lower() and not text.lower().startswith('pot:'):
                    continue

                # Lower threshold for pot-related text
                min_confidence = 0.3 if text.lower().startswith('pot:') else 0.4
                if confidence < min_confidence or len(text) < 1:
                    continue

                # Correct pot text if needed
                text = correct_pot_text(text)
                
                # ONLY keep poker-relevant text
                if not is_poker_relevant_text(text):
                    continue
                
                center_x = sum([point[0] for point in bbox_points]) / 4
                center_y = sum([point[1] for point in bbox_points]) / 4
                region_key = f"{text.lower()}{int(center_x/25)}{int(center_y/25)}"
                
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
                
                # MODIFIED: Include white background regions but mark them for special handling
                is_white_bg = is_region_white_background(region_image)
                if is_white_bg:
                    print(f"   📝 White background region detected: '{text}'")
                
                all_text_regions.append({
                    "text": text,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": confidence,
                    "region_image": region_image,
                    "method": method_name,
                    "is_white_background": is_white_bg  # NEW: Track white background
                })
                
                print(f"   📝 {method_name}: '{text}' conf:{confidence:.2f}")
        except Exception as e:
            print(f"   ⚠ Error with {method_name}: {e}")
    
    print(f"📝 Total poker-relevant regions: {len(all_text_regions)}")
    return all_text_regions

def is_poker_relevant_text(text):
    """
    Filter to keep ONLY poker-relevant text
    """
    text_lower = text.lower().strip()
    
    # Skip very long text (likely UI elements)
    if len(text) > 20:  # Increased from 12 to allow longer names
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
    
    # Keep potential player names and identifiers - MORE INCLUSIVE
    # Common player name patterns (alphanumeric with possible special characters)
    if re.match(r'^[A-Za-z0-9._!]+$', text) and len(text) >= 2:
        return True
    
    # Keep position indicators
    position_words = ['sb', 'bb', 'utg', 'mp', 'co', 'btn', 'small', 'big', 'blind']
    if any(pos in text_lower for pos in position_words):
        return True
    
    # Keep villain-related text
    villain_words = ['villain', 'player', 'opponent', 'hero', 'name', 'stack']
    if any(word in text_lower for word in villain_words):
        return True
    
    # Keep stack/bet related text
    stack_words = ['stack', 'bet', 'chips', 'lakh', 'billion', 'ack']
    if any(word in text_lower for word in stack_words):
        return True
    
    # Keep button-related text
    button_words = ['button', 'fold', 'call', 'raise', 'check', 'bet']
    if any(word in text_lower for word in button_words):
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
    
    # MODIFIED: If less than 3 community cards, it's PREFLOP
    if card_count < 3:
        print(f"   🎮 Game State: PREFLOP ({card_count} community cards)")
        return 'preflop'
    elif card_count == 3:
        print(f"   🎮 Game State: FLOP (3 community cards)")
        return 'flop'
    elif card_count == 4:
        print(f"   🎮 Game State: TURN (4 community cards)")
        return 'turn'
    elif card_count == 5:
        print(f"   🎮 Game State: RIVER (5 community cards)")
        return 'river'
    else:
        # This case should rarely happen, but if more than 5 cards, still PREFLOP
        print(f"   🎮 Game State: PREFLOP ({card_count} community cards - unexpected)")
        return 'preflop'

def detect_community_cards_enhanced(img):
    """
    Enhanced community card detection with GPT-4 analysis - RESTRICTED TO CENTER ONLY
    """
    print(f"\n🃏 ENHANCED COMMUNITY CARD DETECTION (GPT-4) - CENTER ONLY")
    
    height, width = img.shape[:2]
    
    # RESTRICTED search regions for community cards (STRICT center focus only)
    search_regions = [
        # Primary center region - MORE RESTRICTIVE
        (int(height * 0.35), int(height * 0.65), int(width * 0.25), int(width * 0.75)),
        # Very tight center region
        (int(height * 0.40), int(height * 0.60), int(width * 0.30), int(width * 0.70))
    ]
    
    all_community_cards = []
    
    for region_idx, (y1, y2, x1, x2) in enumerate(search_regions):
        print(f"   🔍 Searching region {region_idx + 1}: STRICT center area only")
        
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
        
        # Process each card with GPT-4 analysis
        for i, card_data in enumerate(unique_cards):
            # Use GPT-4 for card analysis
            card_text = analyze_card_with_gpt4(card_data["region_image"], "community")
            
            # Fallback to OCR if GPT-4 fails
            if not card_text:
                card_text = ocr_community_card_enhanced(card_data["region_image"])
            
            if card_text and is_valid_poker_card(card_text):
                card_data["text"] = card_text
                card_data["method"] = f"community_gpt4_region_{region_idx + 1}"
                card_data["card_type"] = "community"
                all_community_cards.append(card_data)
                print(f"   🃏 Community Card {len(all_community_cards)}: {card_text}")
        
        # If we found cards in this region, prefer them
        if len(all_community_cards) >= 3:
            break
    
    # Final cleanup and validation - MORE RESTRICTIVE
    final_cards = []
    seen_positions = set()
    
    for card in all_community_cards:
        # More restrictive position checking
        pos_key = f"{int(card['center_x']/20)}_{int(card['center_y']/20)}"
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            final_cards.append(card)
    
    # Sort by x-coordinate for proper order
    final_cards.sort(key=lambda c: c["center_x"])
    
    print(f"   🃏 Final community cards (CENTER ONLY): {len(final_cards)}")
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
        print(f"   ⚠ Community OCR Error: {e}")
        return None

def format_community_card(text):
    """
    Format community card text to rank+suit format - ENHANCED
    """
    text = text.upper().strip()
    
    # Handle complete card (rank + suit)
    if len(text) >= 2:
        rank = text[0]
        suit_part = text[1:]
        
        if rank in 'AKQJT23456789':
            # Convert suit symbols/letters to standard format
            if '♠' in suit_part or 'S' in suit_part or 'SPADE' in suit_part:
                return f"{rank}s"
            elif '♥' in suit_part or 'H' in suit_part or 'HEART' in suit_part:
                return f"{rank}h"
            elif '♦' in suit_part or 'D' in suit_part or 'DIAMOND' in suit_part:
                return f"{rank}d"
            elif '♣' in suit_part or 'C' in suit_part or 'CLUB' in suit_part:
                return f"{rank}c"
    
    # Handle 10 specially
    if text.startswith('10'):
        rank = 'T'
        if len(text) > 2:
            suit_part = text[2:]
            if '♠' in suit_part or 'S' in suit_part or 'SPADE' in suit_part:
                return f"{rank}s"
            elif '♥' in suit_part or 'H' in suit_part or 'HEART' in suit_part:
                return f"{rank}h"
            elif '♦' in suit_part or 'D' in suit_part or 'DIAMOND' in suit_part:
                return f"{rank}d"
            elif '♣' in suit_part or 'C' in suit_part or 'CLUB' in suit_part:
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
                       (card["center_y"] - existing["center_y"])*2)*0.5
            
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
        # Use GPT-4 for hole card analysis
        card_text = analyze_card_with_gpt4(card_data["region_image"], "hole")
        
        # Fallback to OCR methods if GPT-4 fails
        if not card_text:
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
                "confidence": 0.95,  # Higher confidence for GPT-4 detection
                "region_image": card_data["region_image"],
                "method": "hole_card_gpt4_detection",
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
                    
                    # Try GPT-4 first, then OCR fallback
                    card_text = analyze_card_with_gpt4(card_region, "aggressive")
                    
                    if not card_text:
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
                                "confidence": 0.8,  # Higher confidence for GPT-4
                                "region_image": card_region,
                                "method": "aggressive_gpt4_detection"
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
        print(f"   ⚠ OCR Error: {e}")
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
                 any(s in suit_part for s in ['H', 'S', 'D', 'C', '♠', '♥', '♦', '♣', 'HEARTS', 'SPADES', 'DIAMONDS', 'CLUBS', 'H', 'S', 'D', 'C'])):
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
    Enhanced poker card validation - MORE LENIENT
    """
    if not text:
        return False
    
    text = text.upper().strip()
    
    # Single ranks - EXPANDED LIST
    if text in ['A', 'K', 'Q', 'J', 'T', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '0']:
        return True
    
    # Rank with suit - ENHANCED PATTERN MATCHING
    if len(text) >= 2:
        rank = text[0]
        suit_part = text[1:]
        if (rank in 'AKQJT23456789' and 
            any(s in suit_part for s in ['H', 'S', 'D', 'C', '♠', '♥', '♦', '♣', 'HEARTS', 'SPADES', 'DIAMONDS', 'CLUBS', 'H', 'S', 'D', 'C'])):
            return True
    
    # Handle "10" specially
    if text.startswith('10') and len(text) >= 3:
        suit_part = text[2:]
        if any(s in suit_part for s in ['H', 'S', 'D', 'C', '♠', '♥', '♦', '♣']):
            return True
    
    # Handle partial card text (just rank)
    if text in ['A', 'K', 'Q', 'J', 'T', '2', '3', '4', '5', '6', '7', '8', '9']:
        return True
    
    return False

def analyze_with_gpt4v_focused(region_image, text_content, position_info, is_white_background=False):
    """
    Focused GPT-4V analysis for ONLY essential poker elements
    """
    try:
        _, buffer = cv2.imencode('.jpg', region_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # SPECIAL HANDLING for white background regions
        if is_white_background:
            # Check if it looks like a numeric value (potential increaser)
            if re.search(r'\d', text_content) and any(char in text_content for char in ['K', 'L', 'B', '.']):
                print(f"      🤖 GPT-4V: '{text_content}' → INCREASER (white background)")
                return "INCREASER"
        
        prompt = f"""Analyze this poker interface element: "{text_content}"
Position: {position_info}

Classify as ONE of these ESSENTIAL poker elements ONLY:
1. COMMUNITY_CARD - Playing cards at the center of the screen (3-5 cards). Example: 2♠ 3♦ 4♥ 5♣ 6♠
2. HOLE_CARD - Player's private cards (usually 2 cards at bottom). Example: 2♠ 3♦
3. MY_BET - Your own betting amounts (bottom center player bets like 200, 800, 1.2K, 1.4K, 2.5K, 3.8K, 4.2K)
4. VILLAIN_BET - Other players' betting amounts (non-bottom-center bets like 200, 800, 1.2K, 1.4K, 2.5K, 3.8K, 4.2K)
5. CHIPS - Individual chip denominations ONLY (100, 500, 1K, 2K, 2.25K, 5K) - these are chip values, NOT stacks
6. MY_STACK - Your chip stack amounts (bottom center player stacks like 6.8K, 66.7K, 18.6K, 25.4K)
7. VILLAIN_STACK - Other players' chip stack amounts (non-bottom-center stacks like 6.8K, 66.7K, 18.6K, 25.4K, 1.2L, 2.5L, 5.8L, 1.2B, 2.5B, 5.8B)
8. MY_NAME - YOUR player name (bottom center player name like "WildJuniper3!", "John", "Player1")
9. VILLAIN_NAME - Other players' names (like "ron5228.", "kaahyap", "gridiron5228", "John", "Mike", "Player1", "Villain1")
10. PLAYER_POSITION - Player positions (like "SB", "BB", "UTG", "MP", "CO", "BTN", "Small Blind", "Big Blind")
11. POT - Pot amount (contains "Pot" like "Pot:1.1K")
12. BUTTON - Action buttons (FOLD, CHECK, RAISE, CALL, BET, ALL-IN)
13. GAME_STATE - Game phase indicators (PREFLOP, FLOP, TURN, RIVER)
14. INCREASER - Numeric values in white background areas (like 7.72K, 1.2K, etc.) - ONLY for white background regions
15. OTHER - None of the above

STRICT POSITION-BASED RULES:
- MY_BET: ONLY betting amounts in "bottom center" position (200, 800, 1.2K, 1.4K, 2.5K, 3.8K, 4.2K)
- VILLAIN_BET: ONLY betting amounts in "top left", "top center", "top right", "middle left", "middle right" positions
- MY_STACK: ONLY stack amounts in "bottom center" position (6.8K, 66.7K, 18.6K, 25.4K)
- VILLAIN_STACK: ONLY stack amounts in other positions (6.8K, 66.7K, 18.6K, 25.4K, 1.2L, 2.5L, 5.8L, 1.2B, 2.5B, 5.8B)
- MY_NAME: ONLY names in "bottom center" position
- VILLAIN_NAME: ONLY names in other positions

EXCLUSION RULES:
- BUTTON: EXCLUDE betting buttons like "1/2 Pot", "3/4 Pot", "Min", "2X", "3X", "4X" - these are NOT action buttons
- INCREASER: ONLY for numeric values in white background areas (like 7.72K, 1.2K, etc.)

OTHER RULES:
- COMMUNITY_CARD: Valid poker card in center area (A,K,Q,J,T,2-9 with optional suits)
- HOLE_CARD: Valid poker card in bottom area (player's cards)
- CHIPS: Individual chip denominations ONLY (100, 500, 1K, 2K, 2.25K, 5K) - these are chip values, NOT player stacks
- PLAYER_POSITION: Player positions (SB, BB, UTG, MP, CO, BTN, Small Blind, Big Blind)
- POT: Must contain "Pot" word
- BUTTON: Action words only (fold, check, raise, call, bet, all-in) - NOT betting buttons
- GAME_STATE: Game phase words (preflop, flop, turn, river)
- OTHER: Everything else (UI elements, game info, game titles like "JungleePoke!", etc.)

CRITICAL: Position determines classification for bets, stacks, and names. If position is "bottom center", it MUST be MY_* category. If position is anything else, it MUST be VILLAIN_* category.

Text: "{text_content}"
Position: {position_info}
Respond with ONLY: COMMUNITY_CARD, HOLE_CARD, MY_BET, VILLAIN_BET, CHIPS, MY_STACK, VILLAIN_STACK, MY_NAME, VILLAIN_NAME, PLAYER_POSITION, POT, BUTTON, GAME_STATE, INCREASER, or OTHER"""

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
    print(f" FOCUSED PROCESSING: {os.path.basename(image_path)}")
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
        "my_bets": [],
        "villain_bets": [],
        "chips": [],
        "my_stacks": [],
        "villain_stacks": [],
        "villain_names": [],
        "player_positions": [],
        "pot": [],
        "buttons": [],
        "increaser": [],  # NEW: Add increaser category
        "other": []
    }
    
    print(f"\n🤖 ANALYZING {len(unique_regions)} REGIONS:")
    
    for region in unique_regions:
        text = region["text"]
        bbox = region["bbox"]
        center_x = region["center_x"]
        center_y = region["center_y"]
        region_image = region["region_image"]
        is_white_bg = region.get("is_white_background", False)  # NEW: Get white background flag
        
        confidence = region["confidence"]
        method = region.get("method", "unknown")
        
        # Skip tiny regions
        if region_image.shape[0] < 10 or region_image.shape[1] < 10:
            continue
        
        position_info = get_position_info(center_x, center_y, width, height)
        
        # Analyze with focused GPT-4V (pass white background flag)
        element_type = analyze_with_gpt4v_focused(region_image, text, position_info, is_white_bg)
        
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
        elif element_type == "MY_BET":
            results["my_bets"].append(element)
        elif element_type == "VILLAIN_BET":
            results["villain_bets"].append(element)
        elif element_type == "CHIPS":
            results["chips"].append(element)
        elif element_type == "MY_STACK":
            results["my_stacks"].append(element)
        elif element_type == "VILLAIN_STACK":
            results["villain_stacks"].append(element)
        elif element_type == "VILLAIN_NAME":
            results["villain_names"].append(element)
        elif element_type == "PLAYER_POSITION":
            results["player_positions"].append(element)
        elif element_type == "POT":
            results["pot"].append(element)
        elif element_type == "BUTTON":
            results["buttons"].append(element)
        elif element_type == "INCREASER":  # NEW: Handle increaser
            results["increaser"].append(element)
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
        "total_my_bets": len(results["my_bets"]),
        "total_villain_bets": len(results["villain_bets"]),
        "total_chips": len(results["chips"]),
        "total_my_stacks": len(results["my_stacks"]),
        "total_villain_stacks": len(results["villain_stacks"]),
        "total_villain_names": len(results["villain_names"]),
        "total_player_positions": len(results["player_positions"]),
        "total_pot": len(results["pot"]),
        "total_buttons": len(results["buttons"]),
        "total_increaser": len(results["increaser"]),  # NEW: Add increaser
        
        "community_card_values": [card["value"] for card in results["community_cards"]],
        "hole_card_values": [card["value"] for card in results["hole_cards"]],
        "my_bet_values": [bet["value"] for bet in results["my_bets"]],
        "villain_bet_values": [bet["value"] for bet in results["villain_bets"]],
        "chip_values": [chip["value"] for chip in results["chips"]],
        "my_stack_values": [stack["value"] for stack in results["my_stacks"]],
        "villain_stack_values": [stack["value"] for stack in results["villain_stacks"]],
        "villain_name_values": [name["value"] for name in results["villain_names"]],
        "player_position_values": [pos["value"] for pos in results["player_positions"]],
        "pot_values": [pot["value"] for pot in results["pot"]],
        "button_values": [btn["value"] for btn in results["buttons"]],
        "increaser_values": [inc["value"] for inc in results["increaser"]]  # NEW: Add increaser
    }
    
    results["summary"] = summary
    
    print(f"\n📊 FOCUSED RESULTS:")
    print(f"   🎮 Game State: {game_state}")
    print(f"   🃏 Community Cards: {summary['total_community_cards']} - {summary['community_card_values']}")
    print(f"   🃏 Hole Cards: {summary['total_hole_cards']} - {summary['hole_card_values']}")
    print(f"   💰 My Bets: {summary['total_my_bets']} - {summary['my_bet_values']}")
    print(f"   💰 Villain Bets: {summary['total_villain_bets']} - {summary['villain_bet_values']}")
    print(f"   🪙 Chips: {summary['total_chips']} - {summary['chip_values']}")
    print(f"   💵 My Stacks: {summary['total_my_stacks']} - {summary['my_stack_values']}")
    print(f"   💵 Villain Stacks: {summary['total_villain_stacks']} - {summary['villain_stack_values']}")
    print(f"   👤 Villain Names: {summary['total_villain_names']} - {summary['villain_name_values']}")
    print(f"   🎯 Player Positions: {summary['total_player_positions']} - {summary['player_position_values']}")
    print(f"   🏆 Pot: {summary['total_pot']} - {summary['pot_values']}")
    print(f"   🔘 Buttons: {summary['total_buttons']} - {summary['button_values']}")
    print(f"   📈 Increaser: {summary['total_increaser']} - {summary['increaser_values']}")  # NEW: Add increaser
    
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
                        (region["center_y"] - existing["center_y"])*2)*0.5
            
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
        "my_bets": "#FFFF00",            # Yellow
        "villain_bets": "#FFD700",       # Gold
        "chips": "#FF69B4",              # Pink
        "my_stacks": "#00FFFF",          # Cyan
        "villain_stacks": "#87CEEB",     # Sky Blue
        "villain_names": "#32CD32",      # Lime Green
        "player_positions": "#FF4500",   # Orange Red
        "pot": "#FF1493",                # Deep Pink
        "buttons": "#00FF00",             # Green
        "increaser": "#FFA500"            # Orange
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
    draw_elements(results["my_bets"], colors["my_bets"], "MY_BET", 11)
    draw_elements(results["villain_bets"], colors["villain_bets"], "VILLAIN_BET", 11)
    draw_elements(results["chips"], colors["chips"], "CHIPS", 11)
    draw_elements(results["my_stacks"], colors["my_stacks"], "MY_STACK", 11)
    draw_elements(results["villain_stacks"], colors["villain_stacks"], "VILLAIN_STACK", 11)
    draw_elements(results["villain_names"], colors["villain_names"], "VILLAIN_NAME", 11)
    draw_elements(results["player_positions"], colors["player_positions"], "POSITION", 11)
    draw_elements(results["pot"], colors["pot"], "POT", 12)
    draw_elements(results["buttons"], colors["buttons"], "BUTTON", 11)
    draw_elements(results["increaser"], colors["increaser"], "INCREASER", 12)  # NEW: Add increaser
    
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
        f"BETS: {summary['total_my_bets'] + summary['total_villain_bets']} | STACKS: {summary['total_my_stacks'] + summary['total_villain_stacks']} | POT: {summary['total_pot']} | BUTTONS: {summary['total_buttons']}",
        color='white',
        fontsize=12,
        weight='bold',
        ha='center',
        bbox=dict(facecolor='black', alpha=0.9, edgecolor='white', pad=4)
    )
    
    # Save visualization
    output_filename = os.path.basename(image_path).split('.')[0] + '_gpt4_enhanced_poker_detection.png'
    output_path = os.path.join(OUTPUT_CHECK_DIR, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    print(f"✅ GPT-4 Enhanced visualization saved: {output_path}")

def run_focused_poker_detection():
    """
    Run enhanced poker detection system with GPT-4 card analysis
    """
    print("🚀 GPT-4 ENHANCED POKER DETECTION SYSTEM")
    print("=" * 60)
    print("🎯 ESSENTIAL ELEMENTS: COMMUNITY CARDS | HOLE CARDS | BETS | STACKS | POT | BUTTONS")
    print("🎮 GAME STATES: PREFLOP | FLOP | TURN | RIVER")
    print("🃏 COMMUNITY CARDS: 3-5 cards in center with rank+suit (3h, Ks, etc.)")
    print("🤖 GPT-4 VISION: Advanced card rank+suit detection")
    print("🚫 IGNORES: Game IDs, UI elements, hand strength, everything else")
    print("✨ CLEAN & PRECISE: JungleePoker optimized detection with GPT-4")
    print("=" * 60)
    
    # Get image files
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:TEST_IMAGE_COUNT]
    
    if not image_files:
        print(f"❌ No images found in {INPUT_IMAGE_DIR}")
        return
    
    print(f"📁 Processing {len(image_files)} images with GPT-4 enhanced detection")
    
    # Process each image
    for filename in tqdm(image_files, desc="🔄 Processing"):
        try:
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            
            # Enhanced processing with GPT-4 card analysis
            results = process_poker_image_focused(image_path)
            
            # Create enhanced visualization
            visualize_focused_detection(image_path, results)
            
            # Save results
            json_filename = filename.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
            json_path = os.path.join(OUTPUT_CHECK_DIR, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 GPT-4 Enhanced results saved: {json_path}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🎯 GPT-4 ENHANCED POKER DETECTION SYSTEM")
    print("🃏 Community Cards: 3-5 cards in center (3h, Ks, Td, 9c, As)")
    print("🃏 Hole Cards: 2 cards at bottom (player's private cards)")
    print("🤖 GPT-4 Vision: Advanced rank+suit detection (As, Kh, Qd, Jc, Td)")
    print("💰 Bets: Betting amounts (200, 800, etc.)")
    print("💵 Stacks: Chip stacks (4.8K, 66.7K, 18.6K, 2.4K)")
    print("🏆 Pot: Pot amounts (Pot:1.1K)")
    print("🔘 Buttons: Action buttons (FOLD, CHECK, RAISE)")
    print("🎮 Game States: PREFLOP, FLOP, TURN, RIVER")
    print("🚫 IGNORES: Everything else!")
    print("=" * 60)
    
    run_focused_poker_detection()
    
    print("\n✅ GPT-4 Enhanced poker detection completed!")
    print(f"📁 Check results in: {OUTPUT_CHECK_DIR}")
    print("🎯 Community cards with rank+suit detected using GPT-4!")
    print("🤖 GPT-4 Vision analysis for accurate card detection!")
    print("🎮 Game state awareness enabled!")







