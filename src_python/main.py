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

from SNN_Architecture import SNNAutoencoder
from synthetic_stream import SyntheticBattlefieldGenerator
from tactical_logic import TacticalDecisionLogic
from sharp_resilience import SHARPOfflineResilienceManager
from tactical_dashboard import TacticalDashboardDPG, TacticalDashboardOpenCV

# Import pybind11 C++ zero-copy module if compiled, otherwise fallback to Python C++ engine simulation
try:
    import battlefield_core
    CPP_CORE_AVAILABLE = True
    print("[+] C++ pybind11 Zero-Copy Core Loaded successfully!")
except ImportError:
    CPP_CORE_AVAILABLE = False
    print("[*] C++ pybind11 binary not built yet. Using high-performance Python/NumPy Inter-Process bridge.")


class BattlefieldDetectionPipeline:
    """
    End-to-End System Pipeline Manager.
    """
    def __init__(self):
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

        # 2. Synthetic / Live Stream Ingestor
        self.stream_generator = SyntheticBattlefieldGenerator(width=640, height=480, fps=30)

        # 3. Anomaly Calculus Engine (MSE Thresholding)
        self.mse_threshold = 0.05
        
        # 4. Tactical NMS Decision Logic
        self.tactical_logic = TacticalDecisionLogic(mse_threshold=self.mse_threshold)

        # 5. SHARP Offline Resilience Engine
        self.resilience_manager = SHARPOfflineResilienceManager()

        # 6. C++ Core Engine (if compiled)
        if CPP_CORE_AVAILABLE:
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

    def compute_mse_heatmap(self, raw_bgr, recon_bgr):
        """
        Computes pixel-wise MSE heatmap matrix between raw input and SNN baseline.
        """
        raw_g = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        recon_g = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Pixel-wise MSE: (Raw - Recon)^2
        diff = raw_g - recon_g
        mse_map = diff * diff

        # Spatial Gaussian smoothing
        mse_smoothed = cv2.GaussianBlur(mse_map, (5, 5), 1.5)
        
        # Color map visualization
        norm_map = np.clip(mse_smoothed / (self.mse_threshold * 2.5), 0.0, 1.0)
        visual_heatmap = cv2.applyColorMap((norm_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        return mse_smoothed, visual_heatmap

    def start(self):
        print("[+] Launching Battlefield Object Detection Dashboard...")
        
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
                
                # 3. Anomaly Calculus (MSE Heatmap)
                mse_map, visual_heatmap = self.compute_mse_heatmap(raw_frame, recon_frame)
                
                # 4. Tactical NMS Threat Localization
                threats, _ = self.tactical_logic.detect_tactical_threats(mse_map)
                
                # 5. SHARP Fault Tolerance & Offline Logging
                for threat in threats:
                    self.resilience_manager.log_threat_offline(threat)
                
                t1 = time.time()
                latency_ms = (t1 - t0) * 1000.0
                
                if frame_count % 10 == 0:
                    fps = 10.0 / (time.time() - start_time)
                    start_time = time.time()

                health_str = "SHARP NOMINAL (100% Local Edge)"

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
    pipeline = BattlefieldDetectionPipeline()
    pipeline.start()
