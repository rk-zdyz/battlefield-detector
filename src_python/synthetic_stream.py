"""
synthetic_stream.py
===================
Synthetic Battlefield Multispectral Video Stream Generator.
Simulates terrain backgrounds, dynamic smoke/dust visual noise, thermal speckle,
and camouflaged threat anomalies (armored assets, personnel, disturbed earth).
"""

import cv2
import numpy as np
import time


class SyntheticBattlefieldGenerator:
    """
    Simulates real-time multispectral battlefield video feeds with dynamic visual noise
    and moving camouflaged tactical threats.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_index = 0
        
        # Terrain base grid
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        self.xx, self.yy = np.meshgrid(x, y)
        self.base_terrain = (0.5 + 0.3 * np.sin(self.xx) * np.cos(self.yy)).astype(np.float32)
        
        # Moving threat parameters (Target 1: Camouflaged Vehicle, Target 2: Disturbed Ground)
        self.target1_pos = [100.0, 150.0]
        self.target1_vel = [1.5, 0.8]
        
        self.target2_pos = [450.0, 300.0]
        self.target2_vel = [-1.0, 0.5]
        
        self.smoke_phase = 0.0

    def generate_next_frame(self, enable_noise=True, enable_threats=True):
        self.frame_index += 1
        self.smoke_phase += 0.05
        
        # 1. Base terrain elevation / multispectral background
        terrain = self.base_terrain.copy()
        
        # Add micro terrain texture (foliage/rocks)
        np.random.seed(self.frame_index % 1000)
        micro_texture = np.random.normal(0, 0.03, (self.height, self.width)).astype(np.float32)
        frame = np.clip(terrain + micro_texture, 0.0, 1.0)
        
        # Convert to 3-channel multispectral (Thermal / NIR representation)
        frame_bgr = cv2.merge([
            np.clip(frame * 0.8, 0, 1),
            np.clip(frame * 0.95, 0, 1),
            np.clip(frame * 1.1, 0, 1)
        ])
        frame_uint8 = (frame_bgr * 255.0).astype(np.uint8)
        
        # 2. Inject Camouflaged Threats (Anomalies) & Thermal Hotspots
        if enable_threats:
            # Update target 1 (Hostile Armored Asset with High Thermal Exhaust moving across terrain)
            self.target1_pos[0] = (self.target1_pos[0] + self.target1_vel[0]) % (self.width - 60)
            self.target1_pos[1] = (self.target1_pos[1] + self.target1_vel[1]) % (self.height - 60)
            t1_x, t1_y = int(self.target1_pos[0]), int(self.target1_pos[1])
            
            # Camouflaged threat texture blending
            threat1_roi = frame_uint8[t1_y:t1_y+40, t1_x:t1_x+50]
            if threat1_roi.shape[0] == 40 and threat1_roi.shape[1] == 50:
                # Metallic armor baseline
                cv2.rectangle(frame_uint8, (t1_x, t1_y), (t1_x+50, t1_y+40), (140, 170, 110), -1)
                # Intense thermal engine exhaust (White-hot FLIR signature in R/B channels)
                cv2.circle(frame_uint8, (t1_x+25, t1_y+20), 14, (255, 240, 220), -1)
                cv2.circle(frame_uint8, (t1_x+10, t1_y+10), 6, (255, 255, 255), -1) # Engine hot spot

            # Update target 2 (Concealed personnel / Moving Thermal Infantry)
            self.target2_pos[0] = (self.target2_pos[0] + self.target2_vel[0]) % (self.width - 40)
            self.target2_pos[1] = (self.target2_pos[1] + self.target2_vel[1]) % (self.height - 40)
            t2_x, t2_y = int(self.target2_pos[0]), int(self.target2_pos[1])
            cv2.ellipse(frame_uint8, (t2_x+20, t2_y+20), (25, 15), 30, 0, 360, (230, 210, 190), -1)
            # Body thermal core signature
            cv2.circle(frame_uint8, (t2_x+20, t2_y+20), 8, (250, 250, 255), -1)

            # Target 3 (Static High-Temperature Thermal Exhaust / Hazard)
            cv2.circle(frame_uint8, (520, 120), 18, (240, 230, 255), -1)
            cv2.circle(frame_uint8, (520, 120), 8, (255, 255, 255), -1)

        # 3. Inject Atmospheric Chaotic Visual Noise (Smoke clouds, dust storms, speckle)
        if enable_noise:
            # Rolling smoke clouds
            smoke_x = int((np.sin(self.smoke_phase) + 1.0) * 0.4 * self.width)
            smoke_y = int((np.cos(self.smoke_phase * 0.7) + 1.0) * 0.4 * self.height)
            smoke_overlay = np.zeros_like(frame_uint8)
            cv2.circle(smoke_overlay, (smoke_x + 100, smoke_y + 80), 90, (180, 180, 180), -1)
            cv2.circle(smoke_overlay, (smoke_x + 160, smoke_y + 110), 70, (160, 160, 160), -1)
            smoke_overlay = cv2.GaussianBlur(smoke_overlay, (65, 65), 25)
            
            # Blend smoke transparently
            alpha = 0.45
            frame_uint8 = cv2.addWeighted(frame_uint8, 1.0 - alpha, smoke_overlay, alpha, 0)
            
            # Add thermal sensor speckle noise
            noise_speckle = np.random.randint(-15, 15, frame_uint8.shape, dtype=np.int16)
            noisy = np.clip(frame_uint8.astype(np.int16) + noise_speckle, 0, 255).astype(np.uint8)
            frame_uint8 = noisy

        return frame_uint8


class BattlefieldStreamIngestor:
    """
    Unified Multi-Source Video Stream Ingestor.
    Supports synthetic battlefield generator, live RTSP IP camera feeds,
    USB webcam device indices, and video file inputs (.mp4, .avi).
    """
    def __init__(self, source="synthetic", width=640, height=480, fps=30):
        self.source_str = str(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.synthetic_gen = None

        if self.source_str == "synthetic" or self.source_str == "" or self.source_str == "None":
            self.mode = "synthetic"
            self.synthetic_gen = SyntheticBattlefieldGenerator(width=width, height=height, fps=fps)
            print(f"[Ingestor] Mode: Synthetic Battlefield Stream Generator ({width}x{height} @ {fps} FPS)")
        else:
            self.mode = "video"
            # Attempt to parse integer for webcam device index
            if self.source_str.isdigit():
                source_input = int(self.source_str)
                print(f"[Ingestor] Mode: USB Webcam / Camera Index ({source_input})")
            else:
                source_input = self.source_str
                print(f"[Ingestor] Mode: Live RTSP Stream / Video File ('{source_input}')")

            self.cap = cv2.VideoCapture(source_input)
            if not self.cap.isOpened():
                print(f"[!] Warning: Unable to open video source '{source_input}'. Falling back to Synthetic Stream Generator.")
                self.mode = "synthetic"
                self.synthetic_gen = SyntheticBattlefieldGenerator(width=width, height=height, fps=fps)

    def generate_next_frame(self, enable_noise=True, enable_threats=True):
        if self.mode == "synthetic":
            return self.synthetic_gen.generate_next_frame(enable_noise=enable_noise, enable_threats=enable_threats)
        
        # Read from video file or RTSP camera capture
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Rewind video file if reached EOF
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # If still failing, return black frame
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Resize to pipeline target dimensions
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
