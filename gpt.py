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

def detect_poker_elements(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        if area < 400:
            continue

        if (0.55 < aspect_ratio < 0.85) and (y > height * 0.65):
            elements.append({"type": "hole_card", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        elif (0.55 < aspect_ratio < 0.85) and (height * 0.3 < y < height * 0.5):
            elements.append({"type": "community_card", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        elif (0.9 < aspect_ratio < 6.0) and (y > height * 0.7):
            elements.append({"type": "button", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        elif (1.0 < aspect_ratio < 4.0) and (width * 0.3 < x < width * 0.7) and (y < height * 0.35):
            elements.append({"type": "pot", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        elif (0.4 < aspect_ratio < 5.0):
            elements.append({"type": "stack_or_bet", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

    return elements

def detect_text_with_easyocr(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    game_id_region = (int(width*0.35), 0, int(width*0.65), int(height*0.1))
    pot_region = (int(width*0.4), int(height*0.25), int(width*0.6), int(height*0.35))
    player_bet_region = (0, int(height*0.75), int(width*0.4), height)
    villain_bet_region = (int(width*0.6), int(height*0.7), width, int(height*0.9))

    def crop_with_padding(img, region, padding=5):
        x1, y1, x2, y2 = region
        y1 = max(0, y1 - padding)
        y2 = min(img.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        return img[y1:y2, x1:x2]

    game_id_img = crop_with_padding(img, game_id_region)
    pot_img = crop_with_padding(img, pot_region)
    player_bet_img = crop_with_padding(img, player_bet_region)
    villain_bet_img = crop_with_padding(img, villain_bet_region)

    results = {
        "game_id": "",
        "pot_amount": "0",
        "my_bet": "0",
        "villian_bet": "0"
    }

    game_id_texts = reader.readtext(game_id_img, detail=0, paragraph=True)
    if game_id_texts:
        results["game_id"] = game_id_texts[0].strip()

    pot_texts = reader.readtext(pot_img, detail=1)
    for detection in pot_texts:
        text = detection[1].lower()
        if "pot" in text:
            amount = re.search(r'[\d.,]+[kKlLbB]?', text.split("pot")[-1])
            if amount:
                results["pot_amount"] = format_amount(amount.group())
            break

    player_bet_texts = reader.readtext(player_bet_img, detail=1)
    for detection in player_bet_texts:
        text = detection[1]
        if "🎯" in text:
            amount = re.search(r'[\d.,]+[kKlLbB]?', text.split("🎯")[-1])
            if amount:
                results["my_bet"] = format_amount(amount.group())
            break

    villain_bet_texts = reader.readtext(villain_bet_img, detail=1)
    for detection in villain_bet_texts:
        text = detection[1].strip()
        if text.isdigit():
            results["villian_bet"] = text
            break
        elif re.match(r'^\d+[kKlLbB]?$', text):
            results["villian_bet"] = format_amount(text)
            break
        else:
            amounts = re.findall(r'[\d.,]+[kKlLbB]?', text)
            if amounts:
                results["villian_bet"] = format_amount(amounts[-1])
                break

    return results

def format_amount(raw_amount):
    if not raw_amount:
        return "0"
    try:
        clean_amount = raw_amount.replace(',', '').replace(' ', '')
        if 'k' in clean_amount.lower():
            return str(int(float(clean_amount.lower().replace('k', '')) * 1000))
        elif 'l' in clean_amount.lower():
            return str(int(float(clean_amount.lower().replace('l', '')) * 100000))
        elif 'b' in clean_amount.lower():
            return str(int(float(clean_amount.lower().replace('b', '')) * 1000000000))
        else:
            return str(int(float(clean_amount)))
    except:
        return "0"

def box(x, y, w, h):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

# Additional functions like analyze_with_gpt4, process_poker_image, visualize_detection, run_detection_pipeline
# would follow here unchanged if you already provided them.
# Let me know if you want me to continue this document with the remaining unchanged parts.


def analyze_with_gpt4(image_region, element_type):
    _, buffer = cv2.imencode('.jpg', image_region)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    prompts = {
        "hole_card": """
Return ONLY the poker card in this format: "R s"
R: A,2,3,4,5,6,7,8,9,T,J,Q,K
s: c,d,h,s
If unreadable, return "unknown".
""",
        "community_card": """
Return ONLY the poker card in this format: "R s"
If unreadable, return "unknown".
""",
       "button": """
Return ONLY in this format: "LABEL STATE MYTURN"
LABEL: fold, check, call, bet, raise, all-in (case insensitive)
STATE: active or inactive
MYTURN: yes or no
If any label is unclear, use "unknown".
Examples:
"call active yes"
"raise inactive no"
"""
,
        "pot": """
Return ONLY the pot amount as a number (no symbols).
If it includes 'k' or 'K', write it in thousands, e.g., 1.6k as 1600.
If it includes 'L', write in lakhs, e.g., 1.2L as 120000.
If it includes 'B', write in billions, e.g., 1.1B as 1100000000.
If unreadable, return "0".
""",
        "stack_or_bet": """
Return ONLY in this format: "stack AMOUNT" or "bet AMOUNT"
Examples:
"stack 3200"
"bet 450"
If unreadable, return "unknown 0".
"""
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[element_type]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return "unknown"

def process_poker_image(image_path):
    elements = detect_poker_elements(image_path)
    text_detections = detect_text_with_easyocr(image_path)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    result = {
        "hole_cards": [],
        "community_cards": [],
        "buttons": [],
        "pot_size": text_detections["pot_amount"],
        "player_stacks": [],
        "villain_stacks": [],
        "bets": [
            {"amount": text_detections["my_bet"], "type": "my_bet", "bounding_box": box(0, int(height*0.7), int(width*0.5), height)},
            {"amount": text_detections["villian_bet"], "type": "villian_bet", "bounding_box": box(int(width*0.5), int(height*0.6), width, int(height*0.9))}
        ],
        "game_state": "unknown",
        "game_id": text_detections["game_id"]
    }

    seen_buttons = set()

    for elem in elements:
        analysis = analyze_with_gpt4(elem["image_region"], elem["type"])

        if elem["type"] == "hole_card":
            result["hole_cards"].append({"label": analysis, "bounding_box": elem["bounding_box"]})

        elif elem["type"] == "community_card":
            result["community_cards"].append({"label": analysis, "bounding_box": elem["bounding_box"]})

        elif elem["type"] == "button":
            if analysis in seen_buttons:
                continue
            seen_buttons.add(analysis)
            parts = analysis.split()
            result["buttons"].append({
                "label": parts[0] if len(parts) > 0 else "unknown",
                "state": parts[1] if len(parts) > 1 else "inactive",
                "my_turn": parts[2] if len(parts) > 2 else "no",
                "bounding_box": elem["bounding_box"]
            })

        elif elem["type"] == "pot":
            pass  # Using OCR-detected pot instead

        elif elem["type"] == "stack_or_bet":
            parts = analysis.split()
            if parts[0] == "stack":
                amt = format_amount(parts[1])
                if elem["bounding_box"]["x1"] < width * 0.5:
                    result["player_stacks"].append({"amount": amt, "bounding_box": elem["bounding_box"]})
                else:
                    result["villain_stacks"].append({"amount": amt, "bounding_box": elem["bounding_box"]})
            elif parts[0] == "bet":
                pass  # Using OCR-detected bets instead

    n_community = len(result["community_cards"])
    if n_community == 0:
        result["game_state"] = "preflop"
    elif n_community == 3:
        result["game_state"] = "flop"
    elif n_community == 4:
        result["game_state"] = "turn"
    elif n_community == 5:
        result["game_state"] = "river"

    return result

def visualize_detection(image_path, data):
    img = Image.open(image_path)
    width, height = img.size
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')

    colors = {
        "hole_cards": "#FF0000",
        "community_cards": "#0000FF",
        "buttons": "#00AA00",
        "pot": "#FF00FF",
        "stack": "#FFFF00",
        "bet": "#FFA500",
        "game_id": "#00FFFF",
        "villian_bet": "#FF6347",
        "my_bet": "#7CFC00"
    }

    def draw_boxes(items, color, label_getter):
        for item in items:
            b = item["bounding_box"]
            rect = patches.Rectangle(
                (b["x1"], b["y1"]), b["x2"]-b["x1"], b["y2"]-b["y1"],
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                b["x1"], b["y2"]+15,
                label_getter(item),
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
            )

    draw_boxes(data["hole_cards"], colors["hole_cards"], lambda i: i["label"])
    draw_boxes(data["community_cards"], colors["community_cards"], lambda i: i["label"])
    draw_boxes(data["buttons"], colors["buttons"], lambda i: f"{i['label']} ({i['state']}) {'MY TURN' if i['my_turn']=='yes' else ''}")
    draw_boxes(data["player_stacks"], colors["stack"], lambda i: f"Player Stack {i['amount']}")
    draw_boxes(data["villain_stacks"], colors["stack"], lambda i: f"Villain Stack {i['amount']}")
    
    # Draw pot visualization
    ax.text(
        width * 0.4, height * 0.2, f"Pot: {data['pot_size']}",
        color=colors["pot"], fontsize=14, weight='bold',
        bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
    )
    
    # Draw bets visualization
    for bet in data["bets"]:
        b = bet["bounding_box"]
        if bet["type"] == "my_bet" and bet["amount"] != "0":
            ax.text(
                b["x1"] + 50, b["y1"] + 50, f"🎯 My Bet: {bet['amount']}",
                color=colors["my_bet"], fontsize=12, weight='bold',
                bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
            )
        elif bet["type"] == "villian_bet" and bet["amount"] != "0":
            ax.text(
                b["x1"] + 50, b["y1"] + 50, f"Villain Bet: {bet['amount']}",
                color=colors["villian_bet"], fontsize=12, weight='bold',
                bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
            )

    # Draw game state and ID
    ax.text(
        10, 10, f"Game State: {data['game_state'].upper()}",
        color='white', fontsize=12, weight='bold',
        bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
    )
    
    if data.get("game_id"):
        ax.text(
            width * 0.4, 20, f"Game ID: {data['game_id']}",
            color=colors["game_id"], fontsize=12, weight='bold',
            bbox=dict(facecolor='black', alpha=0.8, edgecolor='white')
        )

    output_filename = os.path.basename(image_path).split('.')[0] + '_analysis.png'
    output_path = os.path.join(OUTPUT_CHECK_DIR, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Analysis saved: {output_path}")

def run_detection_pipeline():
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:TEST_IMAGE_COUNT]

    if not image_files:
        print(f"No images found in {INPUT_IMAGE_DIR}")
        return

    print(f"Analyzing {len(image_files)} images...")

    for filename in tqdm(image_files, desc="Processing"):
        try:
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            analysis = process_poker_image(image_path)
            visualize_detection(image_path, analysis)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    run_detection_pipeline()