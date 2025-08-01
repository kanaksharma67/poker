import os
import json
import base64
import shutil
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
INPUT_IMAGE_DIR = "./poker_screenshots"
DATASET_ROOT = "poker_dataset_yolov8"
CLASS_NAMES = [
    "player_card", "community_card", 
    "fold_button", "check_button", "call_button", "raise_button",
    "pot_text", "villain_stack", "villain_bet", "timer"
]

def setup_folders():
    """Create YOLOv8 dataset structure"""
    folders = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    for folder in folders:
        os.makedirs(os.path.join(DATASET_ROOT, folder), exist_ok=True)

def gpt4v_poker_detector(image_path):
    """Advanced poker analysis with GPT-4 Vision"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this poker table screenshot with extreme precision and return ONLY a valid JSON object.

CRITICAL INSTRUCTIONS:
- Look at the ACTUAL card faces and button text, not just positions
- Read the exact text on buttons (FOLD, CHECK, CALL, RAISE, BET)
- Identify card ranks and suits from the visible card faces
- Pay attention to game state indicators (preflop/flop/turn/river)

DETECTION GUIDELINES:

1. PLAYER CARDS (bottom center, usually highlighted):
   - Look for 2 cards near the bottom center
   - Read the actual card faces (e.g., "Ah" for Ace of Hearts, "Ks" for King of Spades)
   - If cards are face down, use "unknown" for label

2. COMMUNITY CARDS (center table):
   - Count visible community cards: 0=preflop, 3=flop, 4=turn, 5=river
   - Read actual card faces (e.g., "Jc" for Jack of Clubs)
   - Include all visible community cards

3. ACTION BUTTONS (bottom right area):
   - Read the EXACT button text: "FOLD", "CHECK", "CALL", "RAISE", "BET"
   - Note button states: active (highlighted) or inactive (grayed out)
   - Look for betting amount text (e.g., "BET (1.2K)")

4. STACKS AND POT:
   - Read chip amounts from player stacks
   - Find pot amount (usually displayed near center)
   - Look for bet amounts and side pots

5. GAME STATE:
   - Count community cards to determine: preflop(0), flop(3), turn(4), river(5)
   - Look for position indicators (D=Dealer, SB=Small Blind, BB=Big Blind)

Return this exact JSON structure with ACTUAL values:
{
  "player_cards": [
    {"label": "Ah", "x1": 500, "y1": 600, "x2": 550, "y2": 700},
    {"label": "Kd", "x1": 600, "y1": 600, "x2": 650, "y2": 700}
  ],
  "community_cards": [
    {"label": "Jc", "x1": 400, "y1": 300, "x2": 450, "y2": 400},
    {"label": "4h", "x1": 500, "y1": 300, "x2": 550, "y2": 400},
    {"label": "Jd", "x1": 600, "y1": 300, "x2": 650, "y2": 400}
  ],
  "buttons": [
    {"label": "fold", "x1": 300, "y1": 650, "x2": 400, "y2": 700, "state": "active"},
    {"label": "check", "x1": 500, "y1": 650, "x2": 600, "y2": 700, "state": "inactive"},
    {"label": "bet", "x1": 700, "y1": 650, "x2": 800, "y2": 700, "state": "active"}
  ],
  "villain": {"stack": 19800, "current_bet": 0},
  "pot": 106000,
  "game_state": "flop"
}

IMPORTANT: Use "unknown" only if you cannot read the actual card face or button text. Otherwise, provide the exact values you see."""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }],
        max_tokens=2000
    )
    return json.loads(response.choices[0].message.content)

def validate_poker_logic(gpt_data):
    """Ensure poker rules are followed"""
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

def convert_to_yolo(gpt_data, img_size):
    """Convert GPT-4V output to YOLO format"""
    width, height = img_size
    yolo_lines = []
    
    # Player cards (class 0)
    for card in gpt_data["player_cards"]:
        x1, y1, x2, y2 = card["x1"], card["y1"], card["x2"], card["y2"]
        yolo_lines.append(format_yolo_line(0, x1, y1, x2, y2, width, height))
    
    # Community cards (class 1)
    for card in gpt_data["community_cards"]:
        x1, y1, x2, y2 = card["x1"], card["y1"], card["x2"], card["y2"]
        yolo_lines.append(format_yolo_line(1, x1, y1, x2, y2, width, height))
    
    # Buttons (classes 2-5)
    button_classes = {
        "fold": 2, "check": 3, "call": 4, "raise": 5
    }
    for button in gpt_data["buttons"]:
        if button["label"].lower() in button_classes:
            x1, y1, x2, y2 = button["x1"], button["y1"], button["x2"], button["y2"]
            yolo_lines.append(format_yolo_line(button_classes[button["label"].lower()], x1, y1, x2, y2, width, height))
    
    return "\n".join(yolo_lines)

def format_yolo_line(class_id, x1, y1, x2, y2, img_w, img_h):
    """Convert coordinates to YOLO format"""
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_yaml_config():
    """Generate data.yaml for YOLOv8"""
    yaml_content = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASS_NAMES
    }
    with open(os.path.join(DATASET_ROOT, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def process_image(image_path, output_dir, dataset_type):
    """Process single image through the pipeline"""
    try:
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Get GPT-4V analysis
        gpt_data = gpt4v_poker_detector(image_path)
        
        # Validate poker logic
        validate_poker_logic(gpt_data)
        
        # Convert to YOLO format
        yolo_content = convert_to_yolo(gpt_data, (width, height))
        
        # Save files
        filename = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(output_dir, "images", dataset_type, filename))
        
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(output_dir, "labels", dataset_type, label_filename), 'w') as f:
            f.write(yolo_content)
            
    except Exception as e:
        print(f"Skipping {image_path}: {str(e)}")

def main():
    setup_folders()
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split into train/val (80/20)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Process training set
    print("Processing training images...")
    for filename in tqdm(train_files):
        process_image(
            os.path.join(INPUT_IMAGE_DIR, filename),
            DATASET_ROOT,
            'train'
        )
    
    # Process validation set
    print("\nProcessing validation images...")
    for filename in tqdm(val_files):
        process_image(
            os.path.join(INPUT_IMAGE_DIR, filename),
            DATASET_ROOT,
            'val'
        )
    
    # Create YOLOv8 config
    create_yaml_config()
    print(f"\nDataset ready at {DATASET_ROOT}")
    print("Train YOLOv8 with: yolo train data=data.yaml model=yolov8x.pt epochs=500 imgsz=1280")

if __name__ == "__main__":
    main()