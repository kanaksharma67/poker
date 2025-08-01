# Poker Table Analysis with GPT-4 Vision 🃏

This project uses GPT-4 Vision to analyze poker table screenshots and detect cards, buttons, and game elements for automated poker analysis.

## Features / विशेषताएं

- **Card Detection**: Identifies player cards and community cards
- **Button Recognition**: Detects fold, check, call, and raise buttons
- **Stack Analysis**: Reads chip amounts and pot values
- **Game State**: Determines if it's preflop, flop, turn, or river
- **Visualization**: Creates annotated images with bounding boxes

## Setup / सेटअप

### 1. Install Dependencies / डिपेंडेंसी इंस्टॉल करें

```bash
pip install -r requirements.txt
```

### 2. Configure API Key / API की कॉन्फ़िगर करें

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Poker Screenshots / पोकर स्क्रीनशॉट्स जोड़ें

Place your poker screenshots in the `poker_screenshots/` folder.

## Usage / उपयोग

### Check Setup / सेटअप चेक करें
```bash
python check_setup.py
```

### Test Single Image / एक इमेज टेस्ट करें
```bash
python test_gpt.py
```

### Process All Images / सभी इमेजेज प्रोसेस करें
```bash
python gpt.py
```

### Create YOLO Dataset / YOLO डेटासेट बनाएं
```bash
python all.py
```

## Project Structure / प्रोजेक्ट संरचना

```
poker-1/
├── poker_screenshots/     # Input images
├── checkingimg/          # Output with bounding boxes
├── poker-dataset/        # YOLO training dataset
├── gpt.py               # Main analysis script
├── all.py               # YOLO dataset creation
├── test_gpt.py          # Single image testing
├── check_setup.py       # Environment verification
└── requirements.txt     # Dependencies
```

## Troubleshooting / समस्या समाधान

### Common Issues / सामान्य समस्याएं

1. **"GPT-4V returned invalid JSON"**
   - The model sometimes returns malformed JSON
   - Try running `test_gpt.py` to see the raw response
   - Use clearer, higher quality screenshots

2. **"OpenAI API key not found"**
   - Create a `.env` file with your API key
   - Ensure the key is valid and has credits

3. **"No images found"**
   - Add poker screenshots to `poker_screenshots/` folder
   - Supported formats: PNG, JPG, JPEG

### Tips for Better Results / बेहतर परिणामों के लिए टिप्स

- Use high-resolution screenshots
- Ensure good lighting and clear text
- Include the full poker table in the image
- Avoid blurry or compressed images

## Output / आउटपुट

The analysis provides:
- **Player Cards**: Your 2 cards with coordinates
- **Community Cards**: Flop/turn/river cards
- **Buttons**: Action buttons and their states
- **Stacks**: Chip amounts for players
- **Game State**: Current poker round

Results are saved in `checkingimg/` with bounding boxes drawn on the images.

## Advanced Usage / उन्नत उपयोग

### Training Custom YOLO Model / कस्टम YOLO मॉडल ट्रेनिंग

The `all.py` script creates a YOLO dataset for training:
```bash
python all.py
# Then train with:
# yolo train data=data.yaml model=yolov8x.pt epochs=500 imgsz=1280
```

### Custom Analysis / कस्टम विश्लेषण

Modify the prompt in `gpt.py` to focus on specific elements or add new detection features.

## Support / सहायता

If you encounter issues:
1. Run `python check_setup.py` to verify environment
2. Test with `python test_gpt.py` for debugging
3. Check the console output for detailed error messages

---

**Note**: This project requires a valid OpenAI API key with GPT-4 Vision access. Make sure you have sufficient credits for API calls. 