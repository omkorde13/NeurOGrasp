# 🤖⚡ NeurOGrasp — Real Dataset Edition

## Transparent Object Grasping with Spiking Neural Networks + Event Camera

Welcome to NeurOGrasp, a groundbreaking project demonstrating how Spiking Neural Networks (SNNs) can achieve robust transparent object grasping! This notebook guides you through the entire pipeline, from simulating event camera data from real-world images to training and evaluating a sophisticated SNN model.

### Why NeurOGrasp? 🤔

Transparent objects are notoriously challenging for traditional computer vision systems. Event cameras, inspired by biological vision, offer a unique solution by capturing motion-induced 'events' rather than frames. This project harnesses this power, simulating event data from a custom dataset of transparent glass objects to train an SNN capable of accurate grasp detection.

--- 

### 🚀 Getting Started: Your Execution Journey

Follow these steps to explore, train, and evaluate the NeurOGrasp SNN:

1.  **Setup & GPU Check ✅**: Kick things off by running the `Step 1` cells. This will install all necessary libraries and confirm your environment is ready. For optimal performance, a Colab T4 GPU is highly recommended (`Runtime` → `Change runtime type` → `T4`).

2.  **Load Real Dataset (JPG + Pascal VOC XML) 📸📂**: Bring your data to life! Upload your `archive.zip` to Colab. This archive should contain real transparent glass images (`.jpg`) and their corresponding Pascal VOC XML annotations. Run `Step 2` cells to extract and parse this crucial dataset.

    *   **Dataset Link**: Get the dataset from [Kaggle: Transparent Object Detection](https://www.kaggle.com/datasets/dataclusterlabs/transparent-object-detection)

3.  **Dataset Explorer & Statistics 📊🔍**: Dive into your data! `Step 3` cells will help you understand the dataset's characteristics, visualizing sample images with their ground truth bounding boxes and providing key statistics on object counts, sizes, and aspect ratios.

4.  **Event Camera Simulation (v2e) ✨🎥**: Discover the magic of event cameras! `Step 4` demonstrates our custom `v2e` simulator. By applying subtle synthetic motion to static images, we generate event streams – mimicking how a real event camera perceives transparent objects.

5.  **Voxel Grid Encoding 🧠➡️📊**: Neural networks crave structured input. `Step 5` transforms the raw, variable-length event streams into fixed-size `(10, H, W)` voxel grids, ready for SNN processing. These 10 channels capture 5 ON and 5 OFF event bins across time.

6.  **SNN Architecture (LIF + Temporal Attention) 🤯💡**: Unpack the brain behind NeurOGrasp! `Step 6` details and initializes our Spiking Neural Network, featuring biologically plausible LIF neurons, convolutional layers, and a powerful Temporal Attention mechanism that learns to focus on the most informative time-bins of events.

7.  **Loss Functions 🎯⚖️**: Understand how our model learns! `Step 7` defines the `NeurOGraspLoss`, a sophisticated combination of SmoothL1Loss (for precise bounding box and grasp regression) and BCEWithLogitsLoss (for robust confidence and class prediction).

8.  **Training Pipeline ⚙️📈**: Time to train! `Step 8` orchestrates the PyTorch `RealGlassEventDataset`, `DataLoader`, and the complete training loop. The model iteratively learns, optimized with `AdamW` and guided by early stopping to capture the best performing weights.

9.  **Training Curves & Analysis 📉📈**: Visualize the learning journey! `Step 9` plots critical metrics like training/validation loss, giving you clear insights into the model's convergence and performance over epochs.

10. **Inference on Real Test Images 🚀🎯**: Put your trained model to the test! `Step 10` processes unseen test images through the entire pipeline – from event simulation to SNN prediction – demonstrating its real-world application.

11. **Grasp Visualisation 👐✨**: See the grasps in action! `Step 11` visually overlays the model's predicted bounding boxes and precise grasp points onto the original images, offering an intuitive understanding of its capabilities.

12. **Ablation Study — Temporal Attention Contribution 🔬💡**: How important is temporal attention? `Step 12` conducts an ablation study, comparing the full NeurOGrasp model against simplified baselines to quantify the significant impact of the temporal attention mechanism.

13. **Results Summary & Robot Workspace Mapping 🏆🌍**: The grand finale! `Step 13` presents a comprehensive dashboard of performance metrics (mIoU, AP, grasp error) and even maps the predicted grasp points into a hypothetical robot workspace, showcasing the project's practical implications.

---

### ✨ Model Weights: Trained to Perfection ✨

The NeurOGrasp model weights are dynamically trained and optimized during `Step 8: Training Pipeline`. This notebook employs an intelligent early stopping mechanism that automatically saves the `best_wts` (model state dictionary) corresponding to the lowest validation loss achieved. This guarantees that only the highest-performing model is utilized for all subsequent inference and evaluation tasks. You won't need to load any external pre-trained weights to make this notebook shine!
