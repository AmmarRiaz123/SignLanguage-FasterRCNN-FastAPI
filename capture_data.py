import cv2
import os

# Set up the folder path based on our earlier structure
SAVE_DIR = "data/images"
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_images():
    # Ask which letter you want to record
    letter = input("\nEnter the letter you are capturing (A-Z) or type 'quit' to exit: ").upper()
    
    if letter == 'QUIT':
        return False

    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print(f"\n--- Ready to capture: {letter} ---")
    print("Press SPACEBAR to snap a photo.")
    print("Press 'q' if you need to quit early.")

    count = 1
    # Loop until we get the 10 required images for this letter
    while count <= 10: 
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Make a copy of the frame to draw text on (so the saved image stays clean)
        display_frame = frame.copy()
        
        # Show on-screen instructions
        cv2.putText(display_frame, f"Letter: {letter} | Captured: {count-1}/10", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACEBAR to capture, 'q' to quit", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the live feed
        cv2.imshow("Dataset Collector", display_frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # Spacebar key code
            # Format the filename: e.g., A_01.jpg, A_02.jpg
            filename = os.path.join(SAVE_DIR, f"{letter}_{count:02d}.jpg")
            
            # Save the clean frame (without the text)
            cv2.imwrite(filename, frame) 
            print(f"Saved: {filename}")
            count += 1
            
        elif key == ord('q'):  # 'q' key code
            break
            
    # Clean up and close the camera for this round
    cap.release()
    cv2.destroyAllWindows()
    
    if count > 10:
        print(f"Awesome! You got all 10 images for '{letter}'.")
        
    return True

if __name__ == "__main__":
    print("Starting Dataset Collector...")
    # Keep running until the user types 'quit'
    while True:
        if not capture_images():
            break
    print("Data collection complete! Time to annotate.")