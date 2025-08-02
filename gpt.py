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

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

        # Hole cards
        if (0.55 < aspect_ratio < 0.85) and (y > height * 0.65):
            elements.append({"type": "hole_card", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        # Community cards
        elif (0.55 < aspect_ratio < 0.85) and (height * 0.3 < y < height * 0.5):
            elements.append({"type": "community_card", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        # Buttons (more inclusive)
        elif (0.9 < aspect_ratio < 6.0) and (y > height * 0.7):
            elements.append({"type": "button", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        # Pot
        elif (1.0 < aspect_ratio < 4.0) and (width * 0.3 < x < width * 0.7) and (y < height * 0.35):
            elements.append({"type": "pot", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

        # Stack or Bet (more inclusive, catch small labels)
        elif (0.4 < aspect_ratio < 5.0):
            elements.append({"type": "stack_or_bet", "bounding_box": box(x, y, w, h), "image_region": img[y:y+h, x:x+w]})

    return elements


def box(x, y, w, h):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

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
LABEL: FOLD, check, call, bet, raise, all-in
STATE: active, inactive
MYTURN: yes or no
Examples:
"fold active yes"
"check inactive no"
If unreadable, return "unknown inactive no".
""",
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

def format_amount(raw_amount):
    try:
        n = float(raw_amount)
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f}B"
        elif n >= 100_000:
            return f"{n / 100_000:.1f}L"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}k"
        else:
            return str(int(n))
    except:
        return raw_amount

def process_poker_image(image_path):
    elements = detect_poker_elements(image_path)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    result = {
        "hole_cards": [],
        "community_cards": [],
        "buttons": [],
        "pot_size": "0",
        "player_stacks": [],
        "villain_stacks": [],
        "bets": [],
        "game_state": "unknown"
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
            result["pot_size"] = format_amount(analysis)

        elif elem["type"] == "stack_or_bet":
            parts = analysis.split()
            if parts[0] == "stack":
                amt = format_amount(parts[1])
                if elem["bounding_box"]["x1"] < width * 0.5:
                    result["player_stacks"].append({"amount": amt, "bounding_box": elem["bounding_box"]})
                else:
                    result["villain_stacks"].append({"amount": amt, "bounding_box": elem["bounding_box"]})
            elif parts[0] == "bet":
                amt = format_amount(parts[1])
                result["bets"].append({"amount": amt, "bounding_box": elem["bounding_box"]})

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
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')

    colors = {
        "hole_cards": "#FF0000",
        "community_cards": "#0000FF",
        "buttons": "#00AA00",
        "pot": "#FF00FF",
        "stack": "#FFFF00",
        "bet": "#FFA500"
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
    draw_boxes(data["bets"], colors["bet"], lambda i: f"Bet {i['amount']}")

    ax.text(
        10, 10, f"Game State: {data['game_state'].upper()}\nPot: {data['pot_size']}",
        color='white', fontsize=12, weight='bold',
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
            print(f"Error: {e}")

if __name__ == "__main__":
    run_detection_pipeline()
