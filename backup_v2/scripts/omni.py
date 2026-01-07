import argparse
import sys
import os
from typing import Optional

# Add the current directory to sys.path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

# Import our working Qwen2.5-Omni-3B agent
from qwen25_omni_agent import QwenOmniAgent

# --- Configuration ---
MODEL_PATH = "/home/sovthpaw/Models/Qwen2.5-Omni-3B"

def main():
    parser = argparse.ArgumentParser(description="Interact with Qwen2.5-Omni-3B multimodal agent.")
    parser.add_argument("--text", type=str, help="Text prompt for the model.")
    parser.add_argument("--system", type=str, default=None, help="System prompt for the model.")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file.")
    parser.add_argument("--audio", type=str, default=None, help="Path to an audio file.")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file.")

    args = parser.parse_args()

    if not any([args.text, args.image, args.audio, args.video]):
        parser.error("At least one of --text, --image, --audio, or --video must be provided.")

    print("Loading Qwen2.5-Omni-3B multimodal agent...")
    print(f"Model path: {MODEL_PATH}")

    try:
        # Initialize the agent
        agent = QwenOmniAgent(MODEL_PATH)

        # Prepare messages
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        # Create user message with multimodal content
        user_content = []
        if args.text:
            user_content.append({"type": "text", "text": args.text})
        if args.image:
            user_content.append({"type": "image", "image": args.image})
        if args.audio:
            user_content.append({"type": "audio", "audio": args.audio})
        if args.video:
            user_content.append({"type": "video", "video": args.video})

        messages.append({"role": "user", "content": user_content})

        print("\n--- Processing Multimodal Input ---")
        if args.image:
            print(f"Image: {args.image}")
        if args.audio:
            print(f"Audio: {args.audio}")
        if args.video:
            print(f"Video: {args.video}")
        print(f"Text: {args.text}")

        # Generate response
        print("\n--- Generating Response ---")
        response = agent.generate_response(messages)

        print("\n--- Qwen2.5-Omni-3B Response ---")
        print(response)

    except Exception as e:
        print(f"An unexpected error occurred during Qwen2.5-Omni generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()