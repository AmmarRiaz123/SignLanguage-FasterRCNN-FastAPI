import cv2
import os
import string

def get_next_filename(letter):
    for i in range(1, 11):
        filename = f"data/images/{letter}_{i:02d}.jpg"
        if not os.path.exists(filename):
            return filename
    return None

def capture_location_round():
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)
    cap = cv2.VideoCapture(0)
    
    for letter in string.ascii_uppercase:
        filename = get_next_filename(letter)
        if not filename:
            continue
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Letter: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: capture | S: skip | Q: quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                break
            elif key == ord('s') or key == ord('S'):
                break
            elif key == ord('q') or key == ord('Q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
                
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    while True:
        start = input("Start new location round for A-Z? (y/n): ")
        if start.lower() != 'y':
            break
        if not capture_location_round():
            break