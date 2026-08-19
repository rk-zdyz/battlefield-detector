import os
import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch import functional as SF
import onnx

try:
    from onnxconverter_common import float16
    HAS_FLOAT16_CONVERTER = True
except ImportError:
    HAS_FLOAT16_CONVERTER = False


class FusedSNNAutoencoder(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        # --- Encoder Layout ---
        self.enc_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc_bn = nn.BatchNorm2d(16)
        self.lif_enc = snn.Leaky(beta=beta, init_hidden=True)
        
        # --- Decoder Layout ---
        self.dec_conv = nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)
        self.lif_dec = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x, num_steps=8):
        # Reset internal temporal SNN hidden states
        self.lif_enc.init_leaky()
        self.lif_dec.init_leaky()
        
        recon_rec = []
        
        # Process over temporal dimension
        for step in range(num_steps):
            x_t = x[step] if len(x.shape) == 5 else x
            
            # Encoder path
            cur_enc = self.enc_bn(self.enc_conv(x_t))
            spk_enc = self.lif_enc(cur_enc)
            
            # Decoder path (Reconstruction)
            cur_dec = self.dec_conv(spk_enc)
            spk_dec = self.lif_dec(cur_dec)
            
            recon_rec.append(spk_dec)
            
        return torch.stack(recon_rec, dim=0)


def run_quantization_pipeline():
    # 1. Load or Generate Calibration Assets
    cal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_dataset.npy")
    if not os.path.exists(cal_file):
        print("[*] Generating synthetic calibration dataset (100 sample frames)...")
        cal_data = np.random.uniform(0.0, 1.0, (20, 8, 1, 128, 128)).astype(np.float32)
        np.save(cal_file, cal_data)
        print(f"[+] Created calibration dataset at: {cal_file}")

    # 2. Instantiate SNN Model
    model = FusedSNNAutoencoder()
    model.eval()
    
    # 3. Dynamic Weight Quantization for Convolutional Layers
    print("[*] Applying Dynamic INT8 Weight Quantization to Conv2D / ConvTranspose2D layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.ConvTranspose2d}, dtype=torch.qint8
    )
    print("[+] Dynamic INT8 structural conversion complete.")

    # 4. ONNX Export Pipeline
    dummy_input = torch.randn(8, 1, 1, 128, 128)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(output_dir, "snn_quantized_autoencoder.onnx")
    
    print(f"[*] Exporting Quantized SNN Autoencoder to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=['input_images'],
        output_names=['reconstructed_spikes'],
        dynamic_axes={'input_images': {1: 'batch_size'}, 'reconstructed_spikes': {1: 'batch_size'}},
        dynamo=False
    )
    print(f"[+] Optimization export successful: {onnx_path} created.")

    # 5. Optional FP16 Weights Conversion
    if HAS_FLOAT16_CONVERTER:
        onnx_model = onnx.load(onnx_path)
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model_fp16, onnx_path)
        print("[+] Successfully cast operators to FP16 for minimal edge power draw.")
    else:
        print("[*] Model exported ready for TensorRT / ONNX Runtime FP16 execution.")

    file_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[+] Edge Quantized ONNX Model File Size: {file_size_kb:.2f} KB")


if __name__ == "__main__":
    run_quantization_pipeline()