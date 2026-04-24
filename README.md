# NeurOGrasp — Real Dataset Edition

This notebook demonstrates a Spiking Neural Network (SNN) based approach for transparent object grasping, utilizing event camera simulations from real transparent glass images. It involves data loading, event simulation, SNN architecture definition, training, and evaluation.

### We sincerely apologize for not uploading the .ipynb notebook, as GitHub is unable to properly render large notebooks with extensive outputs.
### Execution Steps:

1.  **Setup & GPU Check**: Run `Step 1` cells to install necessary libraries and verify the environment setup, including GPU availability (Colab T4 GPU is recommended).

2.  **Load Real Dataset (JPG + Pascal VOC XML)**: Upload your `archive.zip` file, containing transparent glass images (`.jpg`) and their corresponding Pascal VOC XML annotations, to your Colab environment. Then, run the cells under `Step 2` to extract the dataset and parse the annotations.

3.  **Dataset Explorer & Statistics**: Execute `Step 3` cells to view dataset statistics and visualize sample images with ground truth bounding boxes. This helps in understanding the data distribution.

4.  **Event Camera Simulation (v2e)**: The cells in `Step 4` set up and demonstrate the v2e simulator, which generates event data from static images by simulating subtle motion. This is crucial as event cameras detect change.

5.  **Voxel Grid Encoding**: `Step 5` defines and tests the voxel grid encoder, which transforms variable-length event streams into fixed-size tensors suitable for the SNN. This involves binning events into 10 channels (5 ON, 5 OFF).

6.  **SNN Architecture (LIF + Temporal Attention)**: `Step 6` outlines and initializes the NeurOGrasp SNN model, composed of Spiking Convolutional Blocks, LIF neurons, and a Temporal Attention mechanism to weigh the importance of different time bins.

7.  **Loss Functions**: `Step 7` defines the combined `NeurOGraspLoss` function, which incorporates SmoothL1Loss for bounding box and grasp regression, and BCEWithLogitsLoss for confidence and class prediction.

8.  **Training Pipeline**: Run `Step 8` to set up the PyTorch `RealGlassEventDataset` and `DataLoader`, and then execute the training loop. The model will be trained over several epochs with early stopping based on validation loss.

9.  **Training Curves & Analysis**: `Step 9` visualizes the training and validation loss curves, along with other metrics, providing insights into the model's learning progress.

10. **Inference on Real Test Images**: `Step 10` demonstrates how to perform inference on the held-out test set, processing new images through the event simulation and SNN to generate predictions.

11. **Grasp Visualisation**: `Step 11` provides visual examples of the model's predictions on test images, showing ground truth vs. predicted bounding boxes and grasp points.

12. **Ablation Study — Temporal Attention Contribution**: `Step 12` includes an ablation study to quantify the contribution of the temporal attention mechanism to the model's performance by comparing it against baselines without attention or with only a single time bin.

13. **Results Summary & Robot Workspace Mapping**: `Step 13` presents a comprehensive dashboard of performance metrics and maps the predicted grasp points to a hypothetical robot workspace.

### Model Weights:

The model weights are trained dynamically during the execution of `Step 8: Training Pipeline`. The notebook implements an early stopping mechanism that saves the `best_wts` (model state dictionary) corresponding to the lowest validation loss achieved during training. This ensures that the best performing model from the training run is used for subsequent inference and evaluation steps. No pre-trained weights need to be loaded externally for this notebook to function.

