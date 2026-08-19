 PRD 2: Camouflage Piercer (Anomaly Autoencoder)
1. Objective
Train a Spiking Autoencoder to establish a strict mathematical baseline for standard environments. The model must utilize reconstruction failure to automatically detect and highlight hidden objects, camouflaged targets, and abnormal terrain disruptions. It will operate entirely on event-driven neuromorphic logic to minimize edge power consumption, utilizing fused multispectral data to pierce extreme weather conditions.
2. Data Synthesis & Preprocessing
⚬ Dataset Ingestion (M³FD): Develop a custom PyTorch DataLoader to ingest perfectly aligned RGB and Thermal infrared image pairs from the M³FD multispectral dataset.
⚬ Exclusion Protocol: The training set must be strictly curated to contain absolute zero tactical targets (acting as the "negative baseline" of empty terrain, forests, and roads).
⚬ Normalization & Fusion: Pre-processing must normalize the inputs to handle extreme weather conditions (dust, fog, complete darkness). The pipeline will fuse the RGB and Thermal channels into a single multi-modal tensor before passing it to the network.
⚬ Streaming Efficiency: By leveraging multispectral fused tensors, the network natively filters anomalous temporal patterns in streaming data without needing to cache video locally.
3. Model Architecture
Component	Specification
Topology	Convolutional Spiking Autoencoder featuring an encoder bottleneck (compressing data to a latent vector) and a decoder block for spatial reconstruction.
Neuron Dynamics	Leaky Integrate-and-Fire (LIF) spiking neurons built utilizing snnTorch. These replace standard activations and act as a natural temporal filter optimized over a set number of time steps (T) to filter out chaotic visual noise (e.g., smoke or dust speckle).
Anomaly Calculus (MSE)	Pixel-wise mathematical subtraction: [Expected Reconstruction Output] - [Live Sensor Input].
Dynamic Thresholding	A validation loop must calculate a dynamic MSE threshold based exclusively on normal baseline terrain. Any pixel reconstruction error exceeding this threshold during live inference is mathematically isolated and flagged as a spatial anomaly (representing camouflaged threats or disturbed earth).
Loss Function	Mean Squared Error (MSE) optimized entirely on reconstructing the normal, fused baseline data.
4. Export & Hardware Handoff
⚬ ONNX Export Logic (export_onnx.py): Write explicit torch.onnx.export logic to unroll the temporal simulation loops and convert the heavy PyTorch SNN graph into a highly optimized file named snn_optimized_fp32.onnx.
⚬ Dynamic Axes Setup: Set dynamic axes for batch sizes during the ONNX export. This guarantees that the downstream C++ ONNX Runtime pipeline can seamlessly handle live, asynchronous video frame ingestion without locking or crashing.
⚬ Calibration Deliverable: Generate a 500-sample NumPy array (.npy) of normal fused terrain tensors. The C++ team requires this specific baseline data to safely calibrate the model for INT8 quantization without collapsing network accuracy.