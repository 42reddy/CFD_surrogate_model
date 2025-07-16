CFD Surrogate Modeling with Fourier Neural Operators
Chandramouli Reddy

Overview
This project constructs a physics-informed surrogate model for simulating 2D unsteady fluid flow over
a cylinder using Fourier Neural Operators (FNOs). It demonstrates how operator learning can ap-
proximate PDE solutions efficiently for varying physical parameters like inlet velocity and viscosity.
The trained model parameters can be downloaded from Kaggle at: https://www.kaggle.com/models/reddy42/fourier_neural_operator

Project Structure
• simulation.py — Classical numerical simulation (e.g., finite volume) of the flow field
• data preparation.py — Generates diverse CFD samples by varying inlet velocity and viscosity
• FNO.py — Implements Fourier Neural Operator block and network
• train.py — Trains the model using MSE loss
• eval.py — Evaluates the model with physical metrics such as divergence and vorticity

Methodology
We learn the operator G mapping input fields to the velocity solution u(x, y) = (u, v). The inputs are:
• Binary obstacle mask
• Inlet velocity field (scalar)
• Kinematic viscosity ν

The output is the steady-state velocity vector field u(x, y).
Model Architecture
The model follows the Fourier Neural Operator framework:
• Global convolution in Fourier space using learnable complex-valued filters
• Local convolution in physical space for fine-grained features
• Residual connections and ReLU activations
• Shallow input/output CNN projections

Evaluation Metrics:

Beyond standard Mean Squared Error (MSE), we use:
• Relative L2 Error: Normalized ℓ2 distance between prediction and ground truth
• Divergence Error: Ensures physical consistency: ∇ · u ≈ 0
• Vorticity Error: Evaluates accuracy of curl field

Training

To train the model:
python train.py
Notes
The FNO architecture enables mesh-invariant learning, offering speedups over traditional solvers while main-
taining physical fidelity when evaluated using domain-relevant metrics.

<img width="2370" height="2588" alt="flow_result_x (1)" src="https://github.com/user-attachments/assets/16ea15ff-7fab-41e7-81cc-ed37c743380a" />

