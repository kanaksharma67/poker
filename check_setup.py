import os
import sys
from dotenv import load_dotenv

def check_setup():
    """Check if the environment is properly set up"""
    print("🔍 Checking setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("✅ OpenAI API key found")
        print(f"   Key starts with: {api_key[:10]}...")
    else:
        print("❌ OpenAI API key not found")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    # Check required directories
    required_dirs = ["./poker_screenshots", "./checkingimg"]
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
            # Count images in poker_screenshots
            if dir_path == "./poker_screenshots":
                image_files = [f for f in os.listdir(dir_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"   Found {len(image_files)} image files")
        else:
            print(f"❌ Directory missing: {dir_path}")
            if dir_path == "./poker_screenshots":
                print("   Please add some poker screenshots to this directory")
            else:
                print("   This will be created automatically")
    
    # Check required packages
    required_packages = [
        "openai", "PIL", "tqdm", "python-dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "python-dotenv":
                import dotenv
            else:
                __import__(package)
            print(f"✅ Package installed: {package}")
        except ImportError:
            print(f"❌ Package missing: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n🎉 Setup looks good! You can now run:")
    print("   python gpt.py")
    print("   python test_gpt.py")
    
    return True

if __name__ == "__main__":
    check_setup() 