PRD 1: Spiking Despeckler (SAR Noise Filter)
1. Objective
Build, train, and export a purely neuromorphic spatial low-pass filter. The model must strip severe multiplicative speckle noise from raw Synthetic Aperture Radar (SAR) and optical feeds while perfectly preserving the hard, structural edges of tactical ground targets.
2. Data Synthesis & Preprocessing
⚬ Source Data: High-resolution clean optical and baseline SAR datasets (e.g., MSTAR, VisDrone).
⚬ Corruption Engine: Programmatic Python/OpenCV pipelines that apply dynamic, randomized multiplicative noise (Rayleigh and Gamma distributions) to clean images to simulate harsh battlefield static and sensor degradation.
⚬ Input/Output Mapping: Strict 1:1 paired datasets formatting [Synthetically Corrupted Tensor] -> [Clean Ground-Truth Tensor].
3. Model Architecture
Component	Specification
Topology	Spiking U-Net (S-UNet) featuring down-sampling encoder and up-sampling decoder blocks.
Neuron Dynamics	Leaky Integrate-and-Fire (LIF) neurons utilizing snnTorch or SpikingJelly.
Optimization Mechanics	Surrogate Gradients (e.g., Sigmoid or ATan approximation) to bypass non-differentiable binary step functions during the backward pass.
Readout Mechanism	Accumulation of membrane potential (V_{mem}) at the final timestep (T), decoded back into a continuous 2D pixel array.
Loss Function	Combination of Mean Squared Error (MSE) and Structural Similarity Index (SSIM) to prioritize edge retention.
4. Export & Hardware Handoff
⚬ Format: ONNX format.
⚬ Temporal Unrolling: The export script must statically unroll the spiking timesteps so the graph can be natively ingested by the C++ ONNX Runtime without dynamic loop evaluation.
⚬ Validation: A testing script that asserts the mathematical equivalence (to a tolerance of 10^{-4}) between the native PyTorch output and the ONNX Runtime output.