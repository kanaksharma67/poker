import os
import json
import base64
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
INPUT_IMAGE_DIR = "./poker_screenshots"  # Folder with 2-3 test images
OUTPUT_CHECK_DIR = "./checkingimg"       # Folder for visualized results
os.makedirs(OUTPUT_CHECK_DIR, exist_ok=True)
TEST_IMAGE_COUNT = 3  # Number of test images to process

def gpt4v_poker_detector(image_path):
    """Professional-level poker table analysis with GPT-4 Vision"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this poker table with professional accuracy:

1. CARDS:
- My Cards: 2 cards at bottom center (look for card backs/faces)
- Community Cards: Center table (0 preflop, 3 flop, 1 turn, 1 river)
- Return exact pixel coordinates and card ranks if visible (e.g., Ah, Kd)

2. BUTTONS:
- Fold: Red button with "FOLD" text
- Check/Call: Green button with "CHECK" or "CALL" text
- Raise/Bet: Yellow button with "RAISE" or "BET" text
- Include button state (active/inactive)

3. GAME ELEMENTS:
- Villain stack: Chip count opposite my position
- Villain bet: Current bet amount
- Pot: Total pot value in center
- Game state: preflop/flop/turn/river

Return JSON with:
- All element coordinates (x1,y1,x2,y2)
- Card values, button states, stack amounts
- Game state validation
- Confidence scores (0-100%)

Required Format:
{
  "player_cards": [
    {"label": "Ah", "x1": 500, "y1": 600, "x2": 550, "y2": 700, "confidence": 95},
    {"label": "Kd", "x1": 600, "y1": 600, "x2": 650, "y2": 700, "confidence": 90}
  ],
  "community_cards": [
    {"label": "2h", "x1": 400, "y1": 300, "x2": 450, "y2": 400, "confidence": 85}
  ],
  "buttons": [
    {"label": "fold", "x1": 300, "y1": 650, "x2": 400, "y2": 700, "state": "active", "confidence": 98}
  ],
  "stacks": {
    "villain": {"amount": 12500, "confidence": 90},
    "hero": {"amount": 8700, "confidence": 85}
  },
  "bets": {
    "villain": 1000,
    "pot": 3500
  },
  "game_state": "flop",
  "analysis_confidence": 92
}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"  # Maximum resolution
                    }
                }
            ]
        }],
        max_tokens=3000,
        temperature=0.1
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        if not all(k in result for k in ["player_cards", "community_cards", "game_state"]):
            raise ValueError("Incomplete analysis from GPT-4V")
        return result
    except json.JSONDecodeError:
        raise ValueError("GPT-4V returned invalid JSON")

def validate_poker_logic(gpt_data):
    """Validate poker rules are followed"""
    errors = []
    
    # Game state validation
    expected_community_cards = {
        "preflop": 0,
        "flop": 3,
        "turn": 4,
        "river": 5
    }.get(gpt_data["game_state"], -1)
    
    if expected_community_cards == -1:
        errors.append(f"Invalid game state: {gpt_data['game_state']}")
    elif len(gpt_data["community_cards"]) != expected_community_cards:
        errors.append(f"{gpt_data['game_state']} should have {expected_community_cards} community cards, got {len(gpt_data['community_cards'])}")
    
    # Player must have 2 cards
    if len(gpt_data["player_cards"]) != 2:
        errors.append(f"Expected 2 player cards, got {len(gpt_data['player_cards'])}")
    
    if errors:
        raise ValueError(" | ".join(errors))

def draw_bounding_boxes(image_path, gpt_data):
    """Draw professional-quality bounding boxes and labels"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Element colors and labels
    elements = [
        ("player_cards", "red", "Player Card"),
        ("community_cards", "blue", "Community Card"),
        ("buttons", "green", lambda x: f"{x['label'].title()} Button"),
        ("villain_stack", "cyan", "Villain Stack"),
        ("villain_bet", "pink", "Villain Bet"),
        ("pot_text", "white", "Pot")
    ]
    
    # Draw all detected elements
    for element, color, label in elements:
        for item in gpt_data.get(element, []):
            if isinstance(label, str):
                text = label
            else:
                text = label(item)
            
            # Draw bounding box
            draw.rectangle(
                [item["x1"], item["y1"], item["x2"], item["y2"]],
                outline=color,
                width=3
            )
            
            # Draw label background
            text_width, text_height = draw.textsize(text, font=font)
            draw.rectangle(
                [item["x1"], item["y1"]-text_height-5, item["x1"]+text_width, item["y1"]],
                fill="black"
            )
            
            # Draw label text
            draw.text(
                (item["x1"], item["y1"]-text_height-5),
                text,
                fill=color,
                font=font
            )
    
    # Save visualized image
    output_path = os.path.join(OUTPUT_CHECK_DIR, os.path.basename(image_path))
    img.save(output_path)
    print(f"Saved analyzed image to {output_path}")

def test_gpt4v_labeling():
    """Test GPT-4V on sample images"""
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:TEST_IMAGE_COUNT]
    
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        try:
            # Get professional analysis
            gpt_data = gpt4v_poker_detector(image_path)
            
            # Validate poker logic
            validate_poker_logic(gpt_data)
            
            # Create visualization
            draw_bounding_boxes(image_path, gpt_data)
            
            # Print summary
            print(f"\nAnalysis for {filename}:")
            print(f"Game State: {gpt_data['game_state']}")
            print(f"Player Cards: {[c['label'] for c in gpt_data['player_cards']]}")
            print(f"Community Cards: {[c['label'] for c in gpt_data['community_cards']]}")
            print(f"Active Buttons: {[b['label'] for b in gpt_data['buttons'] if b.get('state') == 'active']}")
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")

if __name__ == "__main__":
    test_gpt4v_labeling()
    print("\nAuto-labeling complete. Check the 'checkingimg' folder for results.")