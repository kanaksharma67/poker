import os
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
import re

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_single_image(image_path):
    """Test GPT-4V on a single image with detailed debugging"""
    print(f"Testing image: {image_path}")
    
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
        
        raw_response = response.choices[0].message.content
        print(f"\nRaw GPT-4V Response:")
        print(f"Length: {len(raw_response)} characters")
        print(f"First 200 chars: {raw_response[:200]}")
        print(f"Last 200 chars: {raw_response[-200:]}")
        
        # Try to clean and parse JSON
        cleaned_response = re.sub(r'```json\s*', '', raw_response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            print(f"\nExtracted JSON string:")
            print(json_str)
            
            # Try to parse
            try:
                result = json.loads(json_str)
                print(f"\n✅ Successfully parsed JSON!")
                print(f"Keys found: {list(result.keys())}")
                return result
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON Parse Error: {e}")
                print(f"Error position: {e.pos}")
                print(f"Line: {e.lineno}, Column: {e.colno}")
                return None
        else:
            print(f"\n❌ No JSON object found in response")
            return None
            
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return None

if __name__ == "__main__":
    # Test with first available image
    input_dir = "./poker_screenshots"
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image = os.path.join(input_dir, image_files[0])
            print(f"Testing with: {test_image}")
            result = test_single_image(test_image)
            
            if result:
                print(f"\n🎉 Test successful! Found {len(result.get('player_cards', []))} player cards")
            else:
                print(f"\n💥 Test failed - check the output above for debugging info")
        else:
            print(f"No images found in {input_dir}")
    else:
        print(f"Directory {input_dir} not found") 