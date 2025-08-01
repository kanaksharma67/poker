import os
import json
from PIL import Image, ImageDraw, ImageFont
import base64
from dotenv import load_dotenv
from openai import OpenAI
import re

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def debug_coordinates(image_path):
    """Debug and visualize the coordinates detected by GPT-4V"""
    print(f"🔍 Debugging coordinates for: {image_path}")
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"📐 Image dimensions: {width}x{height} pixels")
    
    # Get GPT-4V analysis
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
You are analyzing a screenshot from a No Limit Texas Hold'em Poker table.

🎯 YOUR TASK:
Return a JSON object with **extreme precision** containing ONLY what you can clearly detect from the image.

📌 DETECTION RULES:

1. 🃏 PLAYER CARDS:
   - Always two cards, visible at the bottom center.
   - Detect both rank and suit. Use notation like "Ks" = King of Spades, "Ah" = Ace of Hearts.
   - If a card is hidden or unclear, use "unknown".
   - Provide EXACT pixel coordinates for each card's bounding box.

2. 🃏 COMMUNITY CARDS:
   - Located in the center of the table.
   - Use same notation (e.g. "Jc", "9d", etc.).
   - Count visible cards to infer game phase:
     - 0 → preflop
     - 3 → flop
     - 4 → turn
     - 5 → river
   - Provide EXACT pixel coordinates for each card's bounding box.

3. 🟦 BUTTONS (bottom-right zone):
   - Detect exact text: "FOLD", "CHECK", "CALL", "RAISE", "BET"
   - Mark each with its EXACT coordinates and state:
     - "state": "active" if it's clickable/highlighted
     - "state": "inactive" if grayed out or unavailable
   - Provide EXACT pixel coordinates for each button's bounding box.

4. 💰 STACKS & BETS:
   - Detect stack amount (chips) for the main opponent and hero.
   - Detect pot value (center of the table).
   - Read any visible bet amounts from chips or overlay labels.

5. 🎯 COORDINATE SYSTEM:
   - Use pixel coordinates (x1, y1, x2, y2) for bounding boxes
   - x1, y1 = top-left corner
   - x2, y2 = bottom-right corner
   - Measure EXACTLY where each element is located
   - Be precise with button positions and card locations

6. 🎯 OUTPUT STRUCTURE:
Return a single JSON like this:

{
  "player_cards": [
    {"label": "Ah", "x1": 500, "y1": 600, "x2": 550, "y2": 700, "confidence": 95},
    {"label": "Kd", "x1": 600, "y1": 600, "x2": 650, "y2": 700, "confidence": 90}
  ],
  "community_cards": [
    {"label": "Jc", "x1": 400, "y1": 300, "x2": 450, "y2": 400, "confidence": 85},
    {"label": "4h", "x1": 500, "y1": 300, "x2": 550, "y2": 400, "confidence": 85},
    {"label": "Jd", "x1": 600, "y1": 300, "x2": 650, "y2": 400, "confidence": 85}
  ],
  "buttons": [
    {"label": "fold", "x1": 300, "y1": 650, "x2": 400, "y2": 700, "state": "inactive", "confidence": 98},
    {"label": "check", "x1": 500, "y1": 650, "x2": 600, "y2": 700, "state": "inactive", "confidence": 95},
    {"label": "bet", "x1": 700, "y1": 650, "x2": 800, "y2": 700, "state": "active", "confidence": 90}
  ],
  "stacks": {
    "hero": {"amount": 8700, "confidence": 85},
    "villain": {"amount": 19800, "confidence": 90}
  },
  "bets": {
    "villain": 0,
    "pot": 106000
  },
  "game_state": "flop",
  "analysis_confidence": 92
}

🛑 CRITICAL INSTRUCTIONS:
- Provide EXACT pixel coordinates for all bounding boxes
- Measure precisely where each element is located
- Do not guess coordinates - measure them accurately
- Focus on visible elements only. Ignore decorations or avatars.
- Ensure the JSON is valid and parsable.

Respond ONLY with a valid JSON. No explanations. No markdown.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=2000,
            temperature=0.1
        )
        
        response_content = response.choices[0].message.content.strip()
        cleaned_response = re.sub(r'```json\s*', '', response_content)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        json_str = json_match.group(0) if json_match else cleaned_response
        result = json.loads(json_str)
        
        # Debug coordinates
        print(f"\n📊 Coordinate Analysis:")
        print(f"Image size: {width}x{height}")
        
        # Check player cards
        print(f"\n🃏 Player Cards ({len(result.get('player_cards', []))}):")
        for i, card in enumerate(result.get('player_cards', [])):
            coords = f"({card.get('x1', 'N/A')},{card.get('y1', 'N/A')})-({card.get('x2', 'N/A')},{card.get('y2', 'N/A')})"
            size = f"{(card.get('x2', 0) - card.get('x1', 0))}x{(card.get('y2', 0) - card.get('y1', 0))}"
            print(f"  {i+1}. {card.get('label', 'unknown')} at {coords} (size: {size})")
        
        # Check community cards
        print(f"\n🃏 Community Cards ({len(result.get('community_cards', []))}):")
        for i, card in enumerate(result.get('community_cards', [])):
            coords = f"({card.get('x1', 'N/A')},{card.get('y1', 'N/A')})-({card.get('x2', 'N/A')},{card.get('y2', 'N/A')})"
            size = f"{(card.get('x2', 0) - card.get('x1', 0))}x{(card.get('y2', 0) - card.get('y1', 0))}"
            print(f"  {i+1}. {card.get('label', 'unknown')} at {coords} (size: {size})")
        
        # Check buttons
        print(f"\n🟦 Buttons ({len(result.get('buttons', []))}):")
        for i, button in enumerate(result.get('buttons', [])):
            coords = f"({button.get('x1', 'N/A')},{button.get('y1', 'N/A')})-({button.get('x2', 'N/A')},{button.get('y2', 'N/A')})"
            size = f"{(button.get('x2', 0) - button.get('x1', 0))}x{(button.get('y2', 0) - button.get('y1', 0))}"
            state = button.get('state', 'unknown')
            print(f"  {i+1}. {button.get('label', 'unknown')} ({state}) at {coords} (size: {size})")
        
        # Validate coordinates
        print(f"\n✅ Coordinate Validation:")
        valid_coords = True
        
        for element_type, elements in [("player_cards", result.get('player_cards', [])), 
                                     ("community_cards", result.get('community_cards', [])),
                                     ("buttons", result.get('buttons', []))]:
            for i, element in enumerate(elements):
                if all(key in element for key in ['x1', 'y1', 'x2', 'y2']):
                    x1, y1, x2, y2 = element['x1'], element['y1'], element['x2'], element['y2']
                    
                    # Check bounds
                    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        print(f"  ❌ {element_type}[{i}] outside image bounds: ({x1},{y1})-({x2},{y2})")
                        valid_coords = False
                    
                    # Check logic
                    if x2 <= x1 or y2 <= y1:
                        print(f"  ❌ {element_type}[{i}] invalid bounding box: ({x1},{y1})-({x2},{y2})")
                        valid_coords = False
                    
                    # Check reasonable size
                    w, h = x2 - x1, y2 - y1
                    if w < 10 or h < 10 or w > 200 or h > 200:
                        print(f"  ⚠️  {element_type}[{i}] unusual size: {w}x{h}")
        
        if valid_coords:
            print(f"  ✅ All coordinates are valid!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test with first available image
    input_dir = "./poker_screenshots"
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image = os.path.join(input_dir, image_files[0])
            result = debug_coordinates(test_image)
            
            if result:
                print(f"\n🎉 Coordinate debugging complete!")
            else:
                print(f"\n💥 Debugging failed")
        else:
            print(f"No images found in {input_dir}")
    else:
        print(f"Directory {input_dir} not found") 