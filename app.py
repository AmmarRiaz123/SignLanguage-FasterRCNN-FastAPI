"""
Sign Language Detection FastAPI Application

This application provides a web interface for real-time sign language detection
using a trained Faster R-CNN model. Features include:
- MJPEG webcam stream with live inference
- Bounding box visualization on detected signs
- Word builder: accumulates letters when held for multiple frames
- REST API endpoints for inference

Author: Computer Vision Engineer
"""

import os
import time
import threading
from typing import Dict, Optional, Tuple, List
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn


# ==========================================
# Configuration
# ==========================================
MODEL_PATH = "./faster_rcnn_model.pth"  # Path to trained model
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to see detections (tune based on results)
WORD_BUILDER_THRESHOLD = 0.5  # Threshold for word building
WORD_BUILDER_FRAMES = 15  # Number of consecutive frames to confirm a letter
CAMERA_INDEX = 0  # Webcam index (usually 0 for default camera)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Class mapping: 0 = background, 1-26 = A-Z
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IDX_TO_CLASS = {idx + 1: letter for idx, letter in enumerate(ALPHABET)}
NUM_CLASSES = 27


# ==========================================
# Model Loading
# ==========================================
def create_model(num_classes: int = 27) -> nn.Module:
    """
    Create a Faster R-CNN model with the correct number of classes.
    
    Args:
        num_classes: Number of classes (including background)
        
    Returns:
        Faster R-CNN model
    """
    # Load pre-trained model structure (must match training architecture)
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    
    # Modify the box predictor for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = create_model(NUM_CLASSES)
    
    # Load weights if file exists
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Model loaded from: {model_path}")
    else:
        print(f"⚠ Model file not found: {model_path}")
        print("  Running with untrained model (detections will be random)")
    
    model.to(device)
    model.eval()
    
    return model


# Initialize model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model = load_model(MODEL_PATH, DEVICE)


# ==========================================
# Inference Functions
# ==========================================
# ImageNet normalization (MUST match training preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (600, 600)  # Must match training resize


@torch.no_grad()
def run_inference(
    model: nn.Module,
    frame: np.ndarray,
    device: torch.device,
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Run Faster R-CNN inference on a single frame.
    
    Args:
        model: The trained model
        frame: BGR image from OpenCV (H, W, C)
        device: Device to run inference on
        confidence_threshold: Minimum confidence to keep detections
        
    Returns:
        List of detections with boxes scaled back to original frame size
    """
    orig_h, orig_w = frame.shape[:2]
    
    # Convert BGR (OpenCV) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to match training input size (600x600)
    resized_frame = cv2.resize(rgb_frame, INPUT_SIZE)
    
    # Convert to tensor and normalize (matching training preprocessing exactly)
    image_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
    
    # Apply ImageNet normalization (critical - must match training!)
    for c in range(3):
        image_tensor[c] = (image_tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    
    image_tensor = image_tensor.to(device)
    
    # Run inference
    predictions = model([image_tensor])
    pred = predictions[0]
    
    # Debug: print raw prediction stats
    num_raw = len(pred["scores"])
    if num_raw > 0:
        max_score = pred["scores"].max().item()
        print(f"[DEBUG] Raw predictions: {num_raw}, max score: {max_score:.3f}, threshold: {confidence_threshold}")
    
    # Filter predictions by confidence
    keep = pred["scores"] >= confidence_threshold
    
    # Scale factors to map boxes from 600x600 back to original frame size
    scale_x = orig_w / INPUT_SIZE[0]
    scale_y = orig_h / INPUT_SIZE[1]
    
    detections = []
    for i in range(keep.sum().item()):
        idx = torch.where(keep)[0][i]
        box = pred["boxes"][idx].cpu().numpy()
        label = pred["labels"][idx].item()
        score = pred["scores"][idx].item()
        
        # Scale box coordinates back to original frame size
        scaled_box = [
            box[0] * scale_x,  # x1
            box[1] * scale_y,  # y1
            box[2] * scale_x,  # x2
            box[3] * scale_y,  # y2
        ]
        
        detection = {
            "box": scaled_box,
            "label": label,
            "class_name": IDX_TO_CLASS.get(label, f"Class {label}"),
            "score": score
        }
        detections.append(detection)
        
        # Limit to top 5 detections to avoid clutter
        if len(detections) >= 5:
            break
    
    return detections


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame.
    
    Args:
        frame: BGR image
        detections: List of detection dictionaries
        color: Box color in BGR
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        class_name = det["class_name"]
        score = det["score"]
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{class_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label_text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),  # Black text
            2
        )
    
    return annotated


# ==========================================
# Word Builder Logic
# ==========================================
class WordBuilder:
    """
    Builds words from consecutive sign detections.
    
    Logic:
    - Track the most confident detection across frames
    - If the same letter is detected with high confidence for N consecutive frames,
      append it to the current word
    - Provides methods to clear the word or add a space
    """
    
    def __init__(
        self,
        min_frames: int = 15,
        high_confidence_threshold: float = 0.7
    ):
        """
        Args:
            min_frames: Number of consecutive frames required to confirm a letter
            high_confidence_threshold: Minimum confidence for word building
        """
        self.min_frames = min_frames
        self.threshold = high_confidence_threshold
        
        # State
        self.current_word = ""
        self.current_letter = None
        self.consecutive_frames = 0
        self.letter_history = deque(maxlen=30)  # For smoothing
        self.last_added_letter = None
        self.lock = threading.Lock()
    
    def update(self, detections: List[Dict]) -> Optional[str]:
        """
        Update word builder with new detections.
        
        Args:
            detections: List of current frame detections
            
        Returns:
            Newly added letter if any, else None
        """
        with self.lock:
            # Find the most confident detection above threshold
            best_detection = None
            best_score = 0
            
            for det in detections:
                if det["score"] >= self.threshold and det["score"] > best_score:
                    best_detection = det
                    best_score = det["score"]
            
            if best_detection is None:
                # No high-confidence detection, reset
                self.current_letter = None
                self.consecutive_frames = 0
                return None
            
            detected_letter = best_detection["class_name"]
            
            # Track consecutive detections
            if detected_letter == self.current_letter:
                self.consecutive_frames += 1
            else:
                self.current_letter = detected_letter
                self.consecutive_frames = 1
            
            # Check if we should add this letter
            if self.consecutive_frames >= self.min_frames:
                # Avoid adding the same letter repeatedly
                if detected_letter != self.last_added_letter:
                    self.current_word += detected_letter
                    self.last_added_letter = detected_letter
                    self.consecutive_frames = 0  # Reset for next letter
                    return detected_letter
            
            return None
    
    def get_word(self) -> str:
        """Get the current accumulated word."""
        with self.lock:
            return self.current_word
    
    def get_current_letter(self) -> Tuple[Optional[str], int]:
        """Get the currently tracked letter and frame count."""
        with self.lock:
            return self.current_letter, self.consecutive_frames
    
    def clear_word(self):
        """Clear the current word."""
        with self.lock:
            self.current_word = ""
            self.last_added_letter = None
    
    def add_space(self):
        """Add a space to the current word."""
        with self.lock:
            if self.current_word and not self.current_word.endswith(" "):
                self.current_word += " "
                self.last_added_letter = None
    
    def backspace(self):
        """Remove the last character."""
        with self.lock:
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.last_added_letter = None


# Global word builder instance
word_builder = WordBuilder(
    min_frames=WORD_BUILDER_FRAMES,
    high_confidence_threshold=WORD_BUILDER_THRESHOLD
)


# ==========================================
# Video Stream Class
# ==========================================
class VideoCamera:
    """
    Handles webcam capture and frame processing.
    
    Runs inference on each frame and maintains thread-safe access
    to the latest processed frame.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480
    ):
        """
        Args:
            camera_index: Index of the camera (0 for default)
            width: Frame width
            height: Frame height
        """
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.width = width
        self.height = height
        
        # State
        self.current_frame = None
        self.current_detections = []
        self.lock = threading.Lock()
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            print("⚠ Could not open camera. Using placeholder frames.")
    
    def get_frame(self) -> Tuple[bytes, List[Dict]]:
        """
        Capture frame, run inference, and return JPEG bytes.
        
        Returns:
            Tuple of (JPEG bytes, list of detections)
        """
        success, frame = self.camera.read()
        
        if not success:
            # Return a placeholder frame if camera fails
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame, "Camera not available",
                (50, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
        
        # Run inference
        start_time = time.time()
        detections = run_inference(model, frame, DEVICE, CONFIDENCE_THRESHOLD)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update word builder
        word_builder.update(detections)
        
        # Draw detections
        annotated_frame = draw_detections(frame, detections)
        
        # Add overlay information
        self._draw_overlay(annotated_frame, detections, inference_time)
        
        # Store current state
        with self.lock:
            self.current_frame = annotated_frame.copy()
            self.current_detections = detections
        
        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', annotated_frame)
        return jpeg.tobytes(), detections
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        inference_time: float
    ):
        """Draw overlay with current status information."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current word
        word = word_builder.get_word()
        cv2.putText(
            frame, f"Word: {word if word else '(empty)'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        # Current letter being tracked
        current_letter, frames = word_builder.get_current_letter()
        if current_letter:
            progress = min(frames / WORD_BUILDER_FRAMES * 100, 100)
            cv2.putText(
                frame, f"Detecting: {current_letter} ({progress:.0f}%)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        
        # Inference time
        cv2.putText(
            frame, f"FPS: {1000/inference_time:.1f}",
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        # Number of detections
        cv2.putText(
            frame, f"Detections: {len(detections)}",
            (width - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    
    def release(self):
        """Release camera resources."""
        self.camera.release()


# Global camera instance (lazy initialization)
camera: Optional[VideoCamera] = None


def get_camera() -> VideoCamera:
    """Get or create the camera instance."""
    global camera
    if camera is None:
        camera = VideoCamera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    return camera


# ==========================================
# FastAPI Application
# ==========================================
app = FastAPI(
    title="Sign Language Detection API",
    description="Real-time sign language detection using Faster R-CNN",
    version="1.0.0"
)


# ==========================================
# HTML Template (embedded)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(to right, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #888;
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .video-container {
            background: #0f0f23;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .video-header {
            background: linear-gradient(to right, #7b2cbf, #00d4ff);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .video-header h2 {
            font-size: 1.2rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff00;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        #video-feed {
            width: 100%;
            display: block;
            background: #000;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: #0f0f23;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .panel h3 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.1rem;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        
        .word-display {
            font-size: 2rem;
            font-weight: bold;
            min-height: 60px;
            background: #1a1a3e;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 2px solid #333;
            word-wrap: break-word;
        }
        
        .current-letter {
            font-size: 4rem;
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #7b2cbf33, #00d4ff33);
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        .progress-bar {
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(to right, #7b2cbf, #00d4ff);
            transition: width 0.1s linear;
            width: 0%;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .btn-primary {
            background: linear-gradient(to right, #7b2cbf, #00d4ff);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(123, 44, 191, 0.4);
        }
        
        .btn-secondary {
            background: #333;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #444;
        }
        
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c0392b;
        }
        
        .info-panel {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .info-item {
            background: #1a1a3e;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .info-label {
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .instructions {
            background: #1a1a3e;
            padding: 15px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #aaa;
            line-height: 1.6;
        }
        
        .instructions ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sign Language Detection</h1>
            <p class="subtitle">Real-time ASL Alphabet Recognition using Faster R-CNN</p>
        </header>
        
        <div class="main-content">
            <div class="video-container">
                <div class="video-header">
                    <h2>Live Camera Feed</h2>
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span>Live</span>
                    </div>
                </div>
                <img id="video-feed" src="/video_feed" alt="Video Feed">
            </div>
            
            <div class="sidebar">
                <div class="panel">
                    <h3>Current Detection</h3>
                    <div class="current-letter" id="current-letter">-</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <p style="text-align: center; margin-top: 10px; color: #888; font-size: 0.9rem;">
                        Hold sign to add letter
                    </p>
                </div>
                
                <div class="panel">
                    <h3>Built Word</h3>
                    <div class="word-display" id="word-display">-</div>
                    <div class="controls">
                        <button class="btn btn-primary" onclick="addSpace()">Add Space</button>
                        <button class="btn btn-secondary" onclick="backspace()">Backspace</button>
                        <button class="btn btn-danger" onclick="clearWord()">Clear All</button>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Statistics</h3>
                    <div class="info-panel">
                        <div class="info-item">
                            <div class="info-label">Detections</div>
                            <div class="info-value" id="detection-count">0</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Confidence</div>
                            <div class="info-value" id="confidence">-</div>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <h3>Instructions</h3>
                    <div class="instructions">
                        <strong>How to use:</strong>
                        <ul>
                            <li>Show ASL hand signs to the camera</li>
                            <li>Hold a sign steady to add the letter</li>
                            <li>Use "Add Space" between words</li>
                            <li>Signs must be clear and well-lit</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Powered by PyTorch Faster R-CNN | Built with FastAPI</p>
        </footer>
    </div>
    
    <script>
        // Polling interval for status updates (ms)
        const POLL_INTERVAL = 200;
        
        // Poll for current status
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // Update word display
                document.getElementById('word-display').textContent = 
                    data.word || '-';
                
                // Update current letter
                document.getElementById('current-letter').textContent = 
                    data.current_letter || '-';
                
                // Update progress bar
                const progress = (data.consecutive_frames / data.frames_required) * 100;
                document.getElementById('progress-fill').style.width = 
                    Math.min(progress, 100) + '%';
                
                // Update detection count
                document.getElementById('detection-count').textContent = 
                    data.num_detections;
                
                // Update confidence
                if (data.top_confidence > 0) {
                    document.getElementById('confidence').textContent = 
                        (data.top_confidence * 100).toFixed(1) + '%';
                } else {
                    document.getElementById('confidence').textContent = '-';
                }
                
            } catch (error) {
                console.error('Status update failed:', error);
            }
        }
        
        // API calls for word controls
        async function clearWord() {
            await fetch('/clear_word', { method: 'POST' });
            updateStatus();
        }
        
        async function addSpace() {
            await fetch('/add_space', { method: 'POST' });
            updateStatus();
        }
        
        async function backspace() {
            await fetch('/backspace', { method: 'POST' });
            updateStatus();
        }
        
        // Start polling
        setInterval(updateStatus, POLL_INTERVAL);
        updateStatus();
    </script>
</body>
</html>
"""


# ==========================================
# API Routes
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    return HTMLResponse(content=HTML_TEMPLATE)


def generate_frames():
    """
    Generator function for MJPEG streaming.
    
    Yields JPEG frames as multipart data for continuous video streaming.
    This is the core of the MJPEG (Motion JPEG) protocol:
    - Each frame is a complete JPEG image
    - Frames are separated by multipart boundaries
    - Browser displays frames sequentially to create video effect
    """
    cam = get_camera()
    
    while True:
        # Get processed frame as JPEG bytes
        jpeg_bytes, _ = cam.get_frame()
        
        # Yield frame in MJPEG format
        # The boundary separates individual frames
        # Content-Type specifies this frame is a JPEG image
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + 
            jpeg_bytes + 
            b'\r\n'
        )


@app.get("/video_feed")
async def video_feed():
    """
    MJPEG video stream endpoint.
    
    Returns a StreamingResponse with multipart/x-mixed-replace content type.
    The 'boundary=frame' parameter tells the browser where each image ends.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def get_status():
    """
    Get current detection status for UI updates.
    
    Returns:
        JSON with current word, letter being detected, and statistics
    """
    cam = get_camera()
    
    with cam.lock:
        detections = cam.current_detections.copy()
    
    current_letter, consecutive_frames = word_builder.get_current_letter()
    
    # Find top confidence
    top_confidence = 0
    if detections:
        top_confidence = max(d["score"] for d in detections)
    
    return JSONResponse({
        "word": word_builder.get_word(),
        "current_letter": current_letter,
        "consecutive_frames": consecutive_frames,
        "frames_required": WORD_BUILDER_FRAMES,
        "num_detections": len(detections),
        "top_confidence": top_confidence
    })


@app.post("/clear_word")
async def clear_word():
    """Clear the accumulated word."""
    word_builder.clear_word()
    return JSONResponse({"status": "cleared", "word": ""})


@app.post("/add_space")
async def add_space():
    """Add a space to the word."""
    word_builder.add_space()
    return JSONResponse({"status": "space_added", "word": word_builder.get_word()})


@app.post("/backspace")
async def backspace():
    """Remove the last character."""
    word_builder.backspace()
    return JSONResponse({"status": "backspace", "word": word_builder.get_word()})


@app.get("/predict")
async def predict_single(image_path: str, threshold: float = 0.5):
    """
    Run inference on a single image file.
    
    Args:
        image_path: Path to the image file
        threshold: Confidence threshold
        
    Returns:
        JSON with detections
    """
    if not os.path.exists(image_path):
        return JSONResponse(
            {"error": f"Image not found: {image_path}"},
            status_code=404
        )
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        return JSONResponse(
            {"error": "Could not read image"},
            status_code=400
        )
    
    # Run inference
    detections = run_inference(model, frame, DEVICE, threshold)
    
    return JSONResponse({
        "image_path": image_path,
        "num_detections": len(detections),
        "detections": detections
    })


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global camera
    if camera is not None:
        camera.release()
        print("Camera released")


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Sign Language Detection Server")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Word builder threshold: {WORD_BUILDER_THRESHOLD}")
    print(f"Frames for letter confirmation: {WORD_BUILDER_FRAMES}")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
