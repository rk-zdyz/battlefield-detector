import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch import functional as SF
import onnx
from onnxconverter_common import float16


class FusedSNNAutoencoder(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        # --- Encoder Layout ---
        self.enc_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc_bn = nn.BatchNorm2d(16)
        # FP16/INT8 state quantization for the bottleneck activations
        self.q_lif = SF.state_quant(num_bits=8, uniform=True)
        self.lif_enc = snn.Leaky(beta=beta, state_quant=self.q_lif, init_hidden=True)
        
        # --- Decoder Layout ---
        self.dec_conv = nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)
        self.lif_dec = snn.Leaky(beta=beta, state_quant=self.q_lif, init_hidden=True)

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
            spk_enc, mem_enc = self.lif_enc(cur_enc)
            
            # Decoder path (Reconstruction)
            cur_dec = self.dec_conv(spk_enc)
            spk_dec, mem_dec = self.lif_dec(cur_dec)
            
            recon_rec.append(spk_dec)
            
        return torch.stack(recon_rec, dim=0)

if __name__ == "__main__":
    # 1. Load Calibration Assets
    cal_data = np.load("calibration_dataset.npy")
    cal_tensor = torch.tensor(cal_data, dtype=torch.float32)
    
    # 2. Instantiate and Fuse Spatial Layers
    model = FusedSNNAutoencoder()
    model.eval()
    
    # Fuse spatial blocks to prevent redundant edge memory accesses
    model = torch.quantization.fuse_modules(model, [['enc_conv', 'enc_bn']], inplace=True)
    
    # 3. Apply Custom INT8/FP16 Engine Profile
    # Use 'qnnpack' or 'fbgemm' configuration blocks depending on your edge core environment
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    
    # 4. Temporal Calibration Forward Loop
    with torch.no_grad():
        # Iterate over the autoencoder data to determine static dynamic ranges
        for i in range(min(len(cal_tensor), 100)):
            _ = model(cal_tensor[i:i+1])
            
    # Convert tracked layers to specialized 8-bit implementations
    torch.quantization.convert(model, inplace=True)
    print("Static structural conversion complete.")

    # 5. Hybrid Export Pipeline (Targeting ~40% Power Drop)
    # Define an autoencoder dummy input frame tracker: (Steps, Batch, Channel, H, W)
    dummy_input = torch.randn(8, 1, 1, 28, 28)
    
    # Export standard ONNX structure
    torch.onnx.export(
        model,
        dummy_input,
        "snn_autoencoder.onnx",
        export_params=True,
        opset_version=17, # Higher opsets natively optimize structural quantization nodes
        input_names=['input_images'],
        output_names=['reconstructed_spikes'],
        dynamic_axes={'input_images': {1: 'batch_size'}, 'reconstructed_spikes': {1: 'batch_size'}}
    )
    print("Optimization export successful: snn_autoencoder.onnx created.")

# Load the newly exported file
onnx_model = onnx.load("snn_autoencoder.onnx")

# Convert remaining fallback weights/constants to half-precision (FP16)
onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)

# Save the final optimized edge file
onnx.save(onnx_model_fp16, "snn_autoencoder.onnx")
print("Successfully cast leftover operators to FP16 for minimal power draw.")