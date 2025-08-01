import os
import json
import base64
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from openai import OpenAI
import re

# Initialize
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def precise_coordinate_analysis(image_path):
    """Analyze poker table with extreme precision for coordinates"""
    print(f"🎯 Precise coordinate analysis for: {image_path}")
    
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"📐 Image dimensions: {width}x{height} pixels")
    
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
You are a computer vision expert analyzing a poker table screenshot. Your task is to provide EXACT pixel coordinates for all visible elements.

🎯 CRITICAL REQUIREMENTS:
- Measure pixel coordinates with extreme precision
- Use the image as a coordinate grid (0,0 at top-left)
- Provide bounding boxes that tightly fit each element
- Do NOT estimate - measure actual pixel positions

📏 COORDINATE MEASUREMENT INSTRUCTIONS:

1. 🃏 PLAYER CARDS (bottom center area):
   - Find the 2 cards near the bottom center
   - Measure EXACT pixel coordinates for each card's rectangle
   - Include the entire card including any borders/shadows
   - Format: {"label": "Ks", "x1": 1000, "y1": 650, "x2": 1050, "y2": 750}

2. 🃏 COMMUNITY CARDS (center table):
   - Locate cards in the center of the table
   - Measure each visible community card precisely
   - Count cards: 0=preflop, 3=flop, 4=turn, 5=river
   - Include all visible cards with exact coordinates

3. 🟦 ACTION BUTTONS (bottom area):
   - Find buttons like "FOLD", "CHECK", "CALL", "RAISE", "BET"
   - Measure the exact button boundaries
   - Note button state: "active" (clickable) or "inactive" (grayed)
   - Include any text or icons within the button area

4. 💰 STACKS & POT:
   - Detect chip amounts for players
   - Find pot value display
   - Read any visible bet amounts

5. 🎯 MEASUREMENT TECHNIQUE:
   - Use pixel-perfect coordinates
   - x1,y1 = top-left corner of element
   - x2,y2 = bottom-right corner of element
   - Include any borders, shadows, or text within the element
   - Be conservative - better to include a bit extra than miss parts

6. 🎯 OUTPUT FORMAT:
Return ONLY a valid JSON with precise coordinates:

{
  "player_cards": [
    {"label": "Ks", "x1": 1000, "y1": 650, "x2": 1050, "y2": 750, "confidence": 95},
    {"label": "5s", "x1": 1060, "y1": 650, "x2": 1110, "y2": 750, "confidence": 95}
  ],
  "community_cards": [
    {"label": "Jc", "x1": 580, "y1": 340, "x2": 630, "y2": 440, "confidence": 90},
    {"label": "4h", "x1": 640, "y1": 340, "x2": 690, "y2": 440, "confidence": 90},
    {"label": "Jd", "x1": 700, "y1": 340, "x2": 750, "y2": 440, "confidence": 90}
  ],
  "buttons": [
    {"label": "fold", "x1": 900, "y1": 700, "x2": 1000, "y2": 750, "state": "inactive", "confidence": 98},
    {"label": "check", "x1": 1100, "y1": 700, "x2": 1200, "y2": 750, "state": "inactive", "confidence": 95},
    {"label": "bet", "x1": 1300, "y1": 700, "x2": 1400, "y2": 750, "state": "active", "confidence": 90}
  ],
  "stacks": {
    "hero": {"amount": 0, "confidence": 85},
    "villain": {"amount": 19800, "confidence": 90}
  },
  "bets": {
    "villain": 16800,
    "pot": 106000
  },
  "game_state": "flop",
  "analysis_confidence": 92
}

🛑 CRITICAL INSTRUCTIONS:
- Measure coordinates pixel-by-pixel
- Include borders and shadows in bounding boxes
- Be precise - no guessing or estimation
- Ensure all coordinates are within image bounds
- Focus on visible elements only

Respond ONLY with valid JSON. No explanations.
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
        print(f"\n📄 Raw response length: {len(response_content)}")
        print(f"📄 First 200 chars: {response_content[:200]}")
        print(f"📄 Last 200 chars: {response_content[-200:]}")
        
        # Clean response
        cleaned_response = re.sub(r'```json\s*', '', response_content)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            print(f"\n📄 Extracted JSON: {json_str[:200]}...")
        else:
            print(f"\n❌ No JSON object found in response")
            print(f"📄 Full response: {response_content}")
            return None, None, None
        
        try:
            result = json.loads(json_str)
            print(f"✅ JSON parsed successfully!")
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"❌ Error position: {e.pos}")
            print(f"❌ Line: {e.lineno}, Column: {e.colno}")
            print(f"📄 JSON string: {json_str}")
            return None, None, None
        
        return result, width, height
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def visualize_bounding_boxes(image_path, result, width, height):
    """Create a visual debug image with bounding boxes"""
    if not result:
        return
    
    # Load original image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Colors for different elements
    colors = {
        "player_cards": "red",
        "community_cards": "blue", 
        "buttons": "green"
    }
    
    print(f"\n🎨 Visualizing bounding boxes:")
    
    # Draw player cards
    for i, card in enumerate(result.get('player_cards', [])):
        if all(key in card for key in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = card['x1'], card['y1'], card['x2'], card['y2']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-20), f"Player: {card.get('label', 'unknown')}", fill="red", font=font)
            print(f"  🃏 Player card {i+1}: {card.get('label', 'unknown')} at ({x1},{y1})-({x2},{y2})")
    
    # Draw community cards
    for i, card in enumerate(result.get('community_cards', [])):
        if all(key in card for key in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = card['x1'], card['y1'], card['x2'], card['y2']
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text((x1, y1-20), f"Community: {card.get('label', 'unknown')}", fill="blue", font=font)
            print(f"  🃏 Community card {i+1}: {card.get('label', 'unknown')} at ({x1},{y1})-({x2},{y2})")
    
    # Draw buttons
    for i, button in enumerate(result.get('buttons', [])):
        if all(key in button for key in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = button['x1'], button['y1'], button['x2'], button['y2']
            color = "green" if button.get('state') == 'active' else "gray"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1-20), f"Button: {button.get('label', 'unknown')}", fill=color, font=font)
            print(f"  🟦 Button {i+1}: {button.get('label', 'unknown')} ({button.get('state', 'unknown')}) at ({x1},{y1})-({x2},{y2})")
    
    # Save debug image
    debug_path = f"debug_{os.path.basename(image_path)}"
    img.save(debug_path)
    print(f"\n✅ Debug image saved as: {debug_path}")
    
    # Validate coordinates
    print(f"\n✅ Coordinate Validation:")
    valid = True
    
    for element_type, elements in [("player_cards", result.get('player_cards', [])), 
                                 ("community_cards", result.get('community_cards', [])),
                                 ("buttons", result.get('buttons', []))]:
        for i, element in enumerate(elements):
            if all(key in element for key in ['x1', 'y1', 'x2', 'y2']):
                x1, y1, x2, y2 = element['x1'], element['y1'], element['x2'], element['y2']
                
                # Check bounds
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    print(f"  ❌ {element_type}[{i}] outside bounds: ({x1},{y1})-({x2},{y2})")
                    valid = False
                
                # Check logic
                if x2 <= x1 or y2 <= y1:
                    print(f"  ❌ {element_type}[{i}] invalid box: ({x1},{y1})-({x2},{y2})")
                    valid = False
                
                # Check size
                w, h = x2 - x1, y2 - y1
                if w < 5 or h < 5:
                    print(f"  ⚠️  {element_type}[{i}] very small: {w}x{h}")
                elif w > 300 or h > 300:
                    print(f"  ⚠️  {element_type}[{i}] very large: {w}x{h}")
    
    if valid:
        print(f"  ✅ All coordinates are valid!")
    
    return debug_path

if __name__ == "__main__":
    # Test with first available image
    input_dir = "./poker_screenshots"
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image = os.path.join(input_dir, image_files[0])
            result, width, height = precise_coordinate_analysis(test_image)
            
            if result:
                debug_path = visualize_bounding_boxes(test_image, result, width, height)
                print(f"\n🎉 Precise coordinate analysis complete!")
                print(f"📁 Check the debug image: {debug_path}")
            else:
                print(f"\n💥 Analysis failed")
        else:
            print(f"No images found in {input_dir}")
    else:
        print(f"Directory {input_dir} not found") 