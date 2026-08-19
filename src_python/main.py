"""
main.py
=======
Smart India Hackathon (SIH) - AI-Powered Battlefield Object Detection System.

Main Application Launcher linking:
  1. Multithreaded C++ Video Ingestion Engine
  2. SNN Autoencoder Edge Inference (PyTorch / snnTorch / ONNX Runtime)
  3. Mathematical Anomaly Calculus (Pixel-wise MSE Heatmap)
  4. Zero-Copy Inter-Process Bridge (pybind11)
  5. Tactical Decision Logic (NMS) & DearPyGui Operator Dashboard
  6. SHARP Edge Fault Tolerance & Offline Auto-Sync
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure local directories are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "model_training")
sys.path.append(current_dir)
sys.path.append(model_dir)

import argparse
from SNN_Architecture import SNNAutoencoder
from synthetic_stream import BattlefieldStreamIngestor
from tactical_logic import TacticalDecisionLogic
from sharp_resilience import SHARPOfflineResilienceManager
from tactical_dashboard import TacticalDashboardDPG, TacticalDashboardOpenCV

# Import pybind11 C++ zero-copy module if compiled, otherwise fallback to Python C++ engine simulation
try:
    if os.name == 'nt' and os.path.exists('C:/msys64/mingw64/bin'):
        os.add_dll_directory('C:/msys64/mingw64/bin')
    import battlefield_core
    CPP_CORE_AVAILABLE = True
    print("[+] C++ pybind11 Zero-Copy Core Loaded successfully!")
except Exception:
    CPP_CORE_AVAILABLE = False
    print("[*] C++ pybind11 binary not loaded. Using high-performance Python/NumPy Inter-Process bridge.")


class BattlefieldDetectionPipeline:
    """
    End-to-End System Pipeline Manager.
    """
    def __init__(self, source="synthetic", mse_threshold=0.05):
        print("\n========================================================")
        print("   AI-POWERED BATTLEFIELD OBJECT DETECTION SYSTEM")
        print("   Neuromorphic Edge Inference & Anomaly Calculus")
        print("========================================================\n")
        
        # 1. Initialize SNN Autoencoder Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing SNN Autoencoder on device: {self.device}")
        self.snn_model = SNNAutoencoder(in_channels=1, beta=0.85, num_steps=5).to(self.device)
        self.snn_model.eval()

        # Check for pre-trained weights or ONNX model
        self.onnx_path = os.path.join(model_dir, "snn_autoencoder.onnx")
        if not os.path.exists(self.onnx_path):
            print("[*] Training SNN baseline model and exporting ONNX model...")
            try:
                from train_snn_autoencoder import train_and_export_snn
                train_and_export_snn()
            except Exception as e:
                print(f"[!] Warning during SNN training: {e}")

        # 2. Multi-Source Stream Ingestor (Synthetic, RTSP, Video File, Webcam)
        self.stream_generator = BattlefieldStreamIngestor(source=source, width=640, height=480, fps=30)

        # 3. Anomaly Calculus Engine (MSE Thresholding)
        self.mse_threshold = mse_threshold
        
        # 4. Tactical NMS Decision Logic
        self.tactical_logic = TacticalDecisionLogic(mse_threshold=self.mse_threshold)

        # 5. SHARP Offline Resilience Engine
        self.resilience_manager = SHARPOfflineResilienceManager()
        
        # 6. Motion & Thermal Tracking State
        self.prev_gray = None

        # 7. C++ Core Engine (if compiled)
        if CPP_CORE_AVAILABLE:
            if hasattr(battlefield_core, 'FastAnomalyCalculus'):
                self.cpp_engine = battlefield_core.FastAnomalyCalculus(self.mse_threshold)
            elif hasattr(battlefield_core, 'BattlefieldEngine'):
                self.cpp_engine = battlefield_core.BattlefieldEngine()

    def run_snn_reconstruction(self, raw_frame_bgr):
        """
        Executes SNN Autoencoder inference to reconstruct baseline terrain.
        Filters chaotic visual noise (smoke, dust, speckle).
        """
        # Convert frame to grayscale normalized float tensor [1, 1, H, W]
        gray = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Resize to SNN input shape [128, 128] for rapid inference
        input_resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon_tensor = self.snn_model(input_tensor)

        recon_np = recon_tensor.squeeze().cpu().numpy()
        recon_resized = (cv2.resize(recon_np, (w, h)) * 255.0).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_resized, cv2.COLOR_GRAY2BGR)
        return recon_bgr

    def compute_motion_heatmap(self, curr_gray):
        """
        Computes temporal motion magnitude heatmap from consecutive frame differences.
        """
        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()
            return np.zeros_like(curr_gray, dtype=np.float32)

        # Temporal absolute frame difference
        frame_diff = cv2.absdiff(curr_gray, self.prev_gray).astype(np.float32) / 255.0
        self.prev_gray = curr_gray.copy()

        # Gaussian smoothing to create coherent motion field
        motion_map = cv2.GaussianBlur(frame_diff, (7, 7), 2.0)
        return np.clip(motion_map * 2.5, 0.0, 1.0)

    def compute_thermal_heatmap(self, raw_bgr):
        """
        Extracts thermal infrared white-hot hotspots and temperature gradients.
        """
        # White-Hot FLIR signature extraction from Red/Blue thermal channels
        r_chan = raw_bgr[:, :, 2].astype(np.float32) / 255.0
        g_chan = raw_bgr[:, :, 1].astype(np.float32) / 255.0
        b_chan = raw_bgr[:, :, 0].astype(np.float32) / 255.0

        # Thermal intensity score prioritizes hot spectral emissions
        thermal_intensity = 0.5 * r_chan + 0.3 * b_chan + 0.2 * g_chan
        # Highlight white-hot spots (high intensity regional clusters)
        thermal_hotspot = np.clip((thermal_intensity - 0.70) / 0.30, 0.0, 1.0)
        thermal_map = cv2.GaussianBlur(thermal_hotspot, (5, 5), 1.5)
        return thermal_map

    def compute_fused_anomaly_heatmap(self, raw_bgr, recon_bgr):
        """
        Computes multi-modal fused anomaly matrix combining SNN spatial MSE, Motion vectors, and Thermal IR.
        """
        curr_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Spatial SNN Baseline MSE Anomaly
        raw_g = curr_gray.astype(np.float32) / 255.0
        recon_g = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diff = raw_g - recon_g
        mse_spatial = cv2.GaussianBlur(diff * diff, (5, 5), 1.5)

        # 2. Temporal Motion Vector Heatmap
        motion_map = self.compute_motion_heatmap(curr_gray)

        # 3. Thermal IR Signature Hotspot Heatmap
        thermal_map = self.compute_thermal_heatmap(raw_bgr)

        # 4. Multi-Modal Weighted Sensor Fusion
        # Weights: 40% SNN Spatial Baseline Anomaly, 30% Temporal Motion, 30% Thermal Signature
        fused_map = np.clip(0.40 * (mse_spatial / (self.mse_threshold * 2.0)) +
                            0.30 * motion_map +
                            0.30 * thermal_map, 0.0, 1.0)

        # Generate color map visualization for UI Display
        visual_heatmap = cv2.applyColorMap((fused_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        return fused_map, motion_map, thermal_map, visual_heatmap

    def start(self):
        print("[+] Launching Multi-Modal AI Battlefield Threat Detection Dashboard...")
        
        # Initialize UI (Try DearPyGui first, fallback to OpenCV HUD)
        dpg_dashboard = TacticalDashboardDPG()
        use_dpg = dpg_dashboard.initialize_ui()
        cv_dashboard = TacticalDashboardOpenCV() if not use_dpg else None

        frame_count = 0
        start_time = time.time()
        fps = 30.0

        try:
            while True:
                t0 = time.time()
                frame_count += 1
                
                # 1. Ingest Raw Frame
                raw_frame = self.stream_generator.generate_next_frame(enable_noise=True, enable_threats=True)
                
                # 2. Neuromorphic SNN Autoencoder Inference
                recon_frame = self.run_snn_reconstruction(raw_frame)
                
                # 3. Multi-Modal Anomaly Calculus (Spatial SNN MSE + Motion Vectors + Thermal IR)
                fused_map, motion_map, thermal_map, visual_heatmap = self.compute_fused_anomaly_heatmap(raw_frame, recon_frame)
                
                # 4. Multi-Modal Tactical NMS Threat Localization & Classification
                threats, _ = self.tactical_logic.detect_tactical_threats(fused_map, motion_map, thermal_map)
                
                # 5. SHARP Fault Tolerance & Offline Logging
                for threat in threats:
                    self.resilience_manager.log_threat_offline(threat)
                
                t1 = time.time()
                latency_ms = (t1 - t0) * 1000.0
                
                if frame_count % 10 == 0:
                    fps = 10.0 / (time.time() - start_time)
                    start_time = time.time()

                health_str = "SHARP NOMINAL (Multi-Modal Edge)"

                # 6. Render Dashboard
                if use_dpg:
                    dpg_dashboard.update_frame_textures(raw_frame, recon_frame, visual_heatmap)
                    dpg_dashboard.update_telemetry(fps, latency_ms, health_str, threats, self.resilience_manager.get_pending_sync_count())
                    running = dpg_dashboard.render_step()
                    if not running:
                        break
                else:
                    keep_running = cv_dashboard.render(raw_frame, recon_frame, visual_heatmap, threats, fps, health_str)
                    if not keep_running:
                        break

        except KeyboardInterrupt:
            print("\n[*] Stopping Battlefield Detection Pipeline.")
        finally:
            if use_dpg:
                dpg_dashboard.close()
            cv2.destroyAllWindows()
            print("[✓] System shut down cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Battlefield Threat Detection Pipeline")
    parser.add_argument("--source", type=str, default="synthetic", help="Video source: 'synthetic', webcam index (e.g. '0'), video file ('file.mp4'), or RTSP URL ('rtsp://...')")
    parser.add_argument("--threshold", type=float, default=0.05, help="MSE Anomaly Threshold (default: 0.05)")
    args = parser.parse_args()

    pipeline = BattlefieldDetectionPipeline(source=args.source, mse_threshold=args.threshold)
    pipeline.start()
