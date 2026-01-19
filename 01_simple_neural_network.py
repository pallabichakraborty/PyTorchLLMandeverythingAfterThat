import torch
import torch.nn as nn
import torch.optim as optim

# This line ensures that your results are reproducible and consistent every time.
torch.manual_seed(42)

"""
Stage 1 & 2: Data Ingestion and Preparation
Time to prepare your delivery data for training. In the ML pipeline, this combines two stages: Data Ingestion (gathering raw data) and Data Preparation (cleaning it up). In more realistic projects, you'd pull delivery logs from a data source and fix errors or missing values. For this lab, that work is already done, but with a twist. This isn't the same data from the lecture videos. You're looking at a different set of deliveries, which means your model might find a different pattern and make a different prediction for that 7-mile delivery.

Define the two essential tensors for your task:
The distances tensor contains how far you biked for four recent deliveries (in miles).
The times tensor shows how long each delivery took (in minutes).
dtype=torch.float32 sets your data type to 32-bit floating point values for precise calculations.
"""

#Distances in miles for recent bike deliveries
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

#Corresponding delivery times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

"""
Stage 3: Model Building
Now you'll create your model (this is stage 3 of the ML pipeline: Model Building). For bike deliveries, you'll assume a linear relationship between distance and time, a reasonable starting point. Your model will be a single neuron that learns this relationship.

Remember from the lecture videos, a single neuron with one input implements a linear equation:

Time = W × Distance + B

Your job is to find the best values for the weight (W) and bias (B) that fit your delivery data.

Use nn.Sequential(nn.Linear(1, 1)) to create a linear model.
nn.Linear(1, 1): The first 1 means it takes one input (distance), and the second 1 means one neuron that is producing one output (predicted time).
This single linear layer will automatically manage the weight and bias parameters for you.
"""

# Create a model with one input (distance) and one output (time)
model = nn.Sequential(nn.Linear(1, 1))

"""
Stage 4: Training
Time to train your neural network (this is stage 4 of the ML pipeline: Training). You need two key tools to help your model learn from the data:

Loss Function: nn.MSELoss defines the Mean Squared Error loss function.
It measures how wrong your predictions are. If you predict 25 minutes but the actual delivery took 30 minutes, the loss function quantifies that 5-minute error. The model's goal is to minimize this error.
Optimizer: optim.SGD sets up the Stochastic Gradient Descent optimizer. It adjusts your model's weight and bias parameters based on the errors.
lr=0.01: This learning rate controls how big each adjustment step is. Too large and you might overshoot the best values; too small and training takes forever.
"""

# Define the loss function (Mean Squared Error)
loss_function = nn.MSELoss()
# Define the optimizer (Stochastic Gradient Descent) with a learning rate of 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""
Now it's time for your model to learn. The training loop is where your model cycles through the data repeatedly, gradually discovering the relationship between distance and delivery time.

You'll train for 500 epochs (complete passes through your data). During each epoch, these steps occur:

optimizer.zero_grad(): Clears gradients from the previous round. Without this, PyTorch would accumulate adjustments, which could break the learning process.

outputs = model(distances): Performs the "forward pass", where the model makes predictions based on the input distances.

loss = loss_function(outputs, times): Calculates how wrong the predicted outputs are by comparing them to the actual delivery times.

loss.backward(): The "backward pass" (backpropagation) is performed, which calculates exactly how to adjust the weight and bias to reduce the error.

optimizer.step(): Updates the model's parameters using those calculated adjustments.

The loss is printed every 50 epochs to allow you to track the model's learning progress as the error decreases.
"""

# Train the model for 500 epochs
for epoch in range(500):
    # Reset the optimizer's gradients
    optimizer.zero_grad()
    # Make predictions (forward pass)
    outputs = model(distances)
    # Calculate the loss
    loss = loss_function(outputs, times)
    # Calculate adjustments (backward pass)
    loss.backward()
    # Update the model's parameters
    optimizer.step()
    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

"""
Visualizing the Training Results¶
Let's see what your model learned. By plotting the model's predictions as a line against your actual delivery data points, you can check if it found a good pattern.

The helper function, plot_results, will show you:

Your original data points (actual deliveries)
The line your model learned (its predictions)
How well they match
"""
import matplotlib.pyplot as plt
def plot_results(model, distances, times):
    # Generate predictions across a range of distances for a smooth line
    with torch.no_grad():
        predicted_times = model(distances).numpy()
    # Plot actual data points
    plt.scatter(distances.numpy(), times.numpy(), color='red', label='Actual Data')
    # Plot the model's predictions
    plt.plot(distances.numpy(), predicted_times, label='Model Prediction')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (minutes)')
    plt.title('Delivery Time vs Distance')
    plt.legend()
    plt.show()


def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compare the linear model's predictions with actual nonlinear data.
    This function visualizes why a linear model struggles with complex patterns.
    
    Parameters:
    - model: The trained linear model
    - new_distances: Tensor of distances for the combined dataset
    - new_times: Tensor of actual delivery times for the combined dataset
    """
    # Generate predictions using the linear model
    with torch.no_grad():
        linear_predictions = model(new_distances).numpy()
    
    # Convert tensors to numpy for plotting
    distances_np = new_distances.numpy()
    times_np = new_times.numpy()
    
    # Create a figure with better size
    plt.figure(figsize=(12, 6))
    
    # Plot actual data points
    plt.scatter(distances_np, times_np, color='red', s=50, alpha=0.7, 
                label='Actual Data (Bike + Car)', zorder=3)
    
    # Plot the linear model's predictions
    plt.plot(distances_np, linear_predictions, color='blue', linewidth=2, 
             label='Linear Model Prediction', linestyle='--', zorder=2)
    
    # Add a vertical line to show the bike/car transition at 3 miles
    plt.axvline(x=3.0, color='green', linestyle=':', linewidth=2, 
                label='Bike/Car Transition (3 miles)', alpha=0.7)
    
    # Calculate and display the loss
    with torch.no_grad():
        predictions_tensor = model(new_distances)
    loss = nn.MSELoss()(predictions_tensor, new_times)
    
    # Add text box with loss information
    textstr = f'MSE Loss: {loss.item():.2f}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and title
    plt.xlabel('Distance (miles)', fontsize=12)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.title('Linear Model vs. Nonlinear Data (Bike + Car Deliveries)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Loss on combined dataset: {loss.item():.2f}")
    print("\nWhy does the linear model fail?")
    print("- Bike deliveries (≤3 miles): Slower pace, linear relationship")
    print("- Car deliveries (>3 miles): Faster pace, different relationship")
    print("- A single straight line cannot capture both patterns!")
    print("="*60 + "\n")


"""Make Your Prediction
Your model is trained. Now for the moment of truth. Can you make that 7-mile delivery in under 30 minutes?

While a full evaluation would test the model on many unseen data points, here you'll jump straight to its intended purpose: making a data-driven prediction for a specific delivery.

First, you'll set the distance_to_predict variable.
It is initially set to 7.0 to solve the original problem.
After running the code, you can easily come back and change this single variable to get predictions for any other distance.
This variable is then used to create the input tensor for the model.
"""

# Visualize the results before making predictions
plot_results(model, distances, times)

# Distance for which we want to predict the delivery time
distance_to_predict = 7.0

"""
The entire prediction process is wrapped in a with torch.no_grad() block.
This tells PyTorch you're not training anymore, just making a prediction. This makes the process faster and more efficient.
A new input tensor is created using the distance_to_predict variable.
This must be formatted as a 2D tensor ([[7.0]]), as the model expects this specific structure, not a simple number.
Your trained model is called with this new tensor to generate a predicted_time.
After getting the prediction (which is also a tensor), the code extracts the actual numerical value from it using .item().
"""

# Use torch.no_grad() to disable gradient calculations during prediction
with torch.no_grad():
    # Convert the Python variable into a 2D PyTorch tensor that the model expects
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    
    # Pass the new data to the trained model to get a prediction
    predicted_time = model(new_distance)
    
    # Use .item() to extract the scalar value from the tensor for printing
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")

    # Use the scalar value in a conditional statement to make the final decision
    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        print("\nDecision: Take the job. You can make it!")

"""
Inspecting the Model's Learning¶
Now that you have a working model, let's see the exact relationship it learned from the data. You can do this by inspecting the model's internal parameters, the final weight and bias values it discovered during training. These values define the precise line your model is now using to make predictions.
"""
# Access the linear layer from the sequential model
layer = model[0]

# Get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")

"""
Testing Your Model on More Complex Data¶
Your company is expanding its delivery services. To handle longer routes more efficiently, any delivery over 3 miles will now be made by car instead of bike.

That means your dataset just changed. It now includes a mix of bike and car deliveries, two different kinds of trips. You already have a model that worked well before. But will it still work now? Let’s take a closer look.

Define the new dataset, which includes the original bike data plus new data points for longer-distance car deliveries.
new_distances: A tensor containing distances from 1 to 20 miles.
new_times: A tensor with the corresponding delivery times for the combined dataset.
"""
# Combined dataset with bike and car deliveries
new_distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
new_times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

"""
Now let's test how well your bike-only model handles this new mixed dataset.

Use your trained model to generates predictions on the new_distances.
"""
# Use torch.no_grad() to disable gradient calculations during prediction
with torch.no_grad():
    predictions = model(new_distances)

"""
Calculate the new_loss between the model's predictions and the actual times.
Notice how the printed loss value will be significantly higher than the loss at the end of training. This will indicate a poor fit.
"""
new_loss = loss_function(predictions, new_times)
print(f"\nNew Loss on combined dataset: {new_loss.item()}")

"""
To understand why the loss is so high, let's visualize what's happening. 
This plot reveals why your linear model struggles with the new data.
Use the plot_nonlinear_comparison function to see the comparison.
"""
# Visualize the results on the new combined dataset with detailed comparison
plot_nonlinear_comparison(model, new_distances, new_times)

"""
Run the code
source .venv/bin/activate && python -i simple_neural_network.py
"""