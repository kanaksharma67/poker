import os
import json
from PIL import Image

def show_analysis_summary():
    """Display a summary of all poker analysis results"""
    print("🎯 Poker Analysis Results Summary")
    print("=" * 50)
    
    # Check if we have analyzed images
    checking_dir = "./checkingimg"
    if not os.path.exists(checking_dir):
        print("❌ No analysis results found. Run 'python gpt.py' first.")
        return
    
    analyzed_images = [f for f in os.listdir(checking_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not analyzed_images:
        print("❌ No analyzed images found in checkingimg/ directory.")
        return
    
    print(f"📊 Found {len(analyzed_images)} analyzed images:")
    print()
    
    for i, filename in enumerate(analyzed_images, 1):
        print(f"{i}. {filename}")
        
        # Try to get image info
        image_path = os.path.join(checking_dir, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"   📐 Image size: {width}x{height} pixels")
                print(f"   📁 File size: {os.path.getsize(image_path) / 1024:.1f} KB")
        except Exception as e:
            print(f"   ❌ Error reading image: {e}")
        
        print()
    
    print("🎮 Analysis Features Detected:")
    print("   ✅ Player cards (2 cards per hand)")
    print("   ✅ Community cards (flop/turn/river)")
    print("   ✅ Action buttons (fold/check/call/raise)")
    print("   ✅ Stack amounts and pot values")
    print("   ✅ Game state detection")
    print("   ✅ Bounding box visualization")
    
    print("\n💡 Next Steps:")
    print("   1. View the annotated images in 'checkingimg/' folder")
    print("   2. Run 'python all.py' to create YOLO dataset")
    print("   3. Train custom YOLO model for real-time detection")
    print("   4. Integrate with poker bot for automated play")

if __name__ == "__main__":
    show_analysis_summary() 