import  torch
import numpy as np
import pandas as pd

"""
1 - Tensor Creation
The first step in any machine learning pipeline is getting your data ready for the model. In PyTorch, this means loading your data into tensors. You will find that there are several convenient ways to create tensors, whether your data is already in another format or you need to generate it from scratch.

1.1 From Existing Data Structures
Often, your raw data will be in a common format like a Python list, a NumPy array, or a pandas DataFrame. PyTorch provides straightforward functions to convert these structures into tensors, making the data preparation stage more efficient.

torch.tensor(): This function takes input such as a Python list to convert it into a tensor.
Note: The type of numbers you use matters. If you use integers, PyTorch stores them as integers. If you include decimals, they'll be stored as floating point values."""

# From a Python list
x = torch.tensor([1, 2, 3])

print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)

"""
torch.from_numpy(): Converts a NumPy array into a PyTorch tensor."""

# From a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)

print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)

"""From a pandas DataFrame: Pandas is a Python library for working with data organized in rows and columns, like a CSV file or spreadsheet. A DataFrame is pandas' main data structure for storing this kind of tabular data. DataFrames are one of the most common ways to load and explore datasets in machine learning, especially when reading CSV files. There isn't a direct function to convert a DataFrame to a tensor. The standard method is to extract the data from the DataFrame into a NumPy array using the .values attribute, and then convert that array into a tensor using torch.tensor()."""

# From a pandas DataFrame
# Read the data from the CSV file into a DataFrame
df = pd.read_csv('./03_data.csv')

# Extract the data as a NumPy array from the DataFrame
all_values = df.values

# Convert the DataFrame's values to a PyTorch tensor
tensor_from_df = torch.tensor(all_values)

print("ORIGINAL DATAFRAME:\n\n", df)
print("\nRESULTING TENSOR:\n\n", tensor_from_df)
print("\nTENSOR DATA TYPE:", tensor_from_df.dtype)

"""
1.2 - With Predefined Values¶
Sometimes you need to create tensors for specific purposes, like initializing a model's weights and biases before training begins. PyTorch allows you to quickly generate tensors filled with placeholder values like zeros, ones, or random numbers, which is useful for testing and setup.

torch.zeros(): Creates a tensor filled with zeros of the specified dimensions."""

# All zeros tensor
zeros = torch.zeros(2, 3)

print("TENSOR WITH ZEROS:\n\n", zeros)

"""torch.ones(): Creates a tensor filled with ones of the specified dimensions."""

# All ones tensor
ones = torch.ones(2, 3)

print("TENSOR WITH ONES:\n\n", ones)

"""torch.rand(): Generates a tensor with random values uniformly distributed between 0 and 1."""
# Random values tensor
random = torch.rand(2, 3)

print("RANDOM TENSOR:\n\n", random)

"""1.3 - From a Sequence
For situations where you need to generate a sequence of data points, such as a range of values for testing a model's predictions, you can create a tensor directly from that sequence.

torch.arange(): Creates a 1D tensor containing a range of numbers from the specified start value to one less than the specified stop value, incrementing (if positive) or decrementing (if negative) by the specified step value."""
# Tensor from a sequence
range_tensor = torch.arange(start=0, end=10, step=1)
print("A Range Tensor:\n\n", range_tensor)

"""2 - Reshaping & Manipulating
A very common source of errors in PyTorch projects is a mismatch between the shape of your input data and the shape your model expects. For instance, a model is typically designed to process a batch of data, so even if you want to make a single prediction, you must shape your input tensor to look like a batch of one. Mastering tensor reshaping is a key step toward building and debugging models effectively.

2.1 - Checking a Tensor's Dimensions
The first step to fixing a shape mismatch is to understand the current dimensions of your tensor. Checking the shape is your primary debugging tool. It tells you how many samples you have and how many features are in each sample.

torch.Tensor.shape: An attribute that returns a torch.Size object detailing the size of the tensor along each dimension."""
# Create 2D tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)

"""2.2 - Changing a Tensor's Dimensions
Once you identify a shape mismatch, you need to correct it. A frequent task is adding a dimension to a single data sample to create a batch of size one for your model, or removing a dimension after a batch operation is complete.

Adding Dimension: torch.Tensor.unsqueeze() inserts a new dimension at the specified index.
Notice how the shape will change from [2, 3] to [1, 2, 3] and the tensor gets wrapped in an extra pair of square brackets []."""
print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Add dimension
expanded = x.unsqueeze(0)  # Add dimension at index 0

print("\nTENSOR WITH ADDED DIMENSION AT INDEX 0:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)

"""Removing Dimension: torch.Tensor.squeeze() removes dimensions of size 1.
This reverses the unsqueeze operation, removing the 1 from the shape and taking away a pair of outer square brackets."""

print("EXPANDED TENSOR:\n\n", expanded)
print("\nTENSOR SHAPE:", expanded.shape)
print("-"*45)

# Remove dimension
squeezed = expanded.squeeze()

print("\nTENSOR WITH DIMENSION REMOVED:\n\n", squeezed)
print("\nTENSOR SHAPE:", squeezed.shape)

"""2.3 - Restructuring¶
Beyond just adding or removing dimensions, you may need to completely change a tensor's structure to match the requirements of a specific layer or operation within your neural network.

Reshaping: torch.Tensor.reshape() changes the shape of a tensor to the specified dimensions."""
print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Reshape
reshaped = x.reshape(3, 2)

print("\nAFTER PERFORMING reshape(3, 2):\n\n", reshaped)
print("\nTENSOR SHAPE:", reshaped.shape)


"""Transposing: torch.Tensor.transpose() swaps the specified dimensions of a tensor."""
print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)
print("-"*45)

# Transpose
transposed = x.transpose(0, 1)

print("\nAFTER PERFORMING transpose(0, 1):\n\n", transposed)
print("\nTENSOR SHAPE:", transposed.shape)

"""2.4 - Combining Tensors
In the data preparation stage, you might need to combine data from different sources or merge separate batches into one larger dataset.

torch.cat(): Joins a sequence of tensors along an existing dimension. Note: All tensors must have the same shape in dimensions other than the one being concatenated."""
# Create two tensors
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])

# Concatenate along columns (dim=1)
concatenated_tensors = torch.cat((tensor_a, tensor_b), dim=1)


print("TENSOR A:\n\n", tensor_a)
print("\nTENSOR B:\n\n", tensor_b)
print("-"*45)
print("\nCONCATENATED TENSOR (dim=1):\n\n", concatenated_tensors)

"""3 - Indexing & Slicing
After you have your data in a tensor, you will often need to access specific parts of it. Whether you are grabbing a single prediction to inspect its value, separating your input features from your labels, or selecting a subset of data for analysis, indexing and slicing are the tools for the job.

3.1 - Accessing Elements
These are the fundamental techniques for getting data out of a tensor, working very similarly to how you would access elements in a standard Python list.

Standard Indexing: Accessing single elements or entire rows using integer indices (e.g., x[0], x[1, 2])."""
# Create 3x4 tensor
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get a single element at row 1, column 2
single_element_tensor = x[1, 2]

print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)
print("-" * 55)

# Get the entire second row (index 1)
second_row = x[1]

print("\nINDEXING ENTIRE ROW [1]:", second_row)
print("-" * 55)

# Last row
last_row = x[-1]

print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row, "\n")

"""Slicing: Extracting sub-tensors using [start:end:step] notation (e.g., x[:2, ::2]).
Note: The end index itself is not included in the slice.
Slicing can be used to access entire columns."""
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get the first two rows
first_two_rows = x[0:2]

print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)
print("-" * 55)

# Get the third column of all rows
third_column = x[:, 2]

print("\nSLICING THIRD COLUMN ([:, 2]]):", third_column)
print("-" * 55)

# Every other column
every_other_col = x[:, ::2]

print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)
print("-" * 55)

# Last column
last_col = x[:, -1]

print("\nLAST COLUMN ([:, -1]):", last_col, "\n")

"""Combining Indexing & Slicing: You can mix indexing and slicing to access specific sub-tensors."""
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Combining slicing and indexing (First two rows, last two columns)
combined = x[0:2, 2:]

print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined, "\n")

""".item(): Extracts the value from a single-element tensor as a standard Python number."""
print("SINGLE-ELEMENT TENSOR:", single_element_tensor)
print("-" * 45)

# Extract the value from a single-element tensor as a standard Python number
value = single_element_tensor.item()

print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))


"""3.2 - Advanced Indexing
For more complex data selection, such as filtering your dataset based on one or more conditions, you can use advanced indexing techniques.

Boolean Masking: Using a boolean tensor to select elements that meet a certain condition (e.g., x[x > 5])."""
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Boolean indexing using logical comparisons
mask = x > 6

print("MASK (VALUES > 6):\n\n", mask, "\n")

# Applying Boolean masking
mask_applied = x[mask]

print("VALUES AFTER APPLYING MASK:", mask_applied, "\n")

"""
Fancy Indexing: Using a tensor of indices to select specific elements in a non-contiguous way."""
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Fancy indexing

# Get first and third rows
row_indices = torch.tensor([0, 2])

# Get second and fourth columns
col_indices = torch.tensor([1, 3]) 

# Gets values at (0,1), (0,3), (2,1), (2,3)
get_values = x[row_indices[:, None], col_indices]

print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values, "\n")

"""4 - Mathematical & Logical Operations
At their core, neural networks are performing mathematical computations. A single neuron, for example, calculates a weighted sum of its inputs and adds a bias. PyTorch is optimized to perform these operations efficiently across entire tensors at once, which is what makes training so fast.

4.1 - Arithmetic
These operations are the foundation of how a neural network processes data. You'll see how PyTorch handles element-wise calculations and uses a powerful feature called broadcasting to simplify your code.

Element-wise Operations: Standard math operators (+, *) that apply to each element independently."""
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 60)

# Element-wise addition
element_add = a + b

print("\nAFTER PERFORMING ELEMENT-WISE ADDITION:", element_add, "\n")

print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

# Element-wise multiplication
element_mul = a * b

print("\nAFTER PERFORMING ELEMENT-WISE MULTIPLICATION:", element_mul, "\n")

"""Dot Product (torch.matmul()): Calculates the dot product of two vectors or matrices."""
print("TENSOR A:", a)
print("TENSOR B", b)
print("-" * 65)

# Dot product
dot_product = torch.matmul(a, b)

print("\nAFTER PERFORMING DOT PRODUCT:", dot_product, "\n")

"""Broadcasting: The automatic expansion of smaller tensors to match the shape of larger tensors during arithmetic operations.
Broadcasting allows operations between tensors with compatible shapes, even if they don't have the exact same dimensions."""

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1],
                 [2],
                 [3]])

print("TENSOR A:", a)
print("SHAPE:", a.shape)
print("\nTENSOR B\n\n", b)
print("\nSHAPE:", b.shape)
print("-" * 65)

# Apply broadcasting
c = a + b


print("\nTENSOR C:\n\n", c)
print("\nSHAPE:", c.shape, "\n")

"""4.2 - Logic & Comparisons
Logical operations are powerful tools for data preparation and analysis. They allow you to create boolean masks to filter, select, or modify your data based on specific conditions you define.

Comparison Operators: Element-wise comparisons (>, ==, <) that produce a boolean tensor."""
temperatures = torch.tensor([20, 35, 19, 35, 42])
print("TEMPERATURES:", temperatures)
print("-" * 50)

### Comparison Operators (>, <, ==)

# Use '>' (greater than) to find temperatures above 30
is_hot = temperatures > 30

# Use '<=' (less than or equal to) to find temperatures 20 or below
is_cool = temperatures <= 20

# Use '==' (equal to) to find temperatures exactly equal to 35
is_35_degrees = temperatures == 35

print("\nHOT (> 30 DEGREES):", is_hot)
print("COOL (<= 20 DEGREES):", is_cool)
print("EXACTLY 35 DEGREES:", is_35_degrees, "\n")

"""Logical Operators: Element-wise logical operations (& for AND, | for OR) on boolean tensors."""
is_morning = torch.tensor([True, False, False, True])
is_raining = torch.tensor([False, False, True, True])
print("IS MORNING:", is_morning)
print("IS RAINING:", is_raining)
print("-" * 50)

### Logical Operators (&, |)

# Use '&' (AND) to find when it's both morning and raining
morning_and_raining = (is_morning & is_raining)

# Use '|' (OR) to find when it's either morning or raining
morning_or_raining = is_morning | is_raining

print("\nMORNING & (AND) RAINING:", morning_and_raining)
print("MORNING | (OR) RAINING:", morning_or_raining)

"""4.3 - Statistics
Calculating statistics like the mean or standard deviation can be useful for understanding your dataset or for implementing certain types of normalization during the data preparation phase.

torch.mean(): Calculates the mean of all elements in a tensor."""

data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
print("DATA:", data)
print("-" * 45)

# Calculate the mean
data_mean = data.mean()

print("\nCALCULATED MEAN:", data_mean, "\n")

"""torch.std(): Calculates the standard deviation of all elements."""
print("DATA:", data)
print("-" * 45)

# Calculate the standard deviation
data_std = data.std()

print("\nCALCULATED STD:", data_std, "\n")

"""4.4 - Data Types
Just as important as a tensor's shape is its data type. Neural networks typically perform their calculations using 32-bit floating point numbers (float32). Providing data of the wrong type, such as an integer, can lead to runtime errors or unexpected behavior during training. It is a good practice to ensure your tensors have the correct data type for your model.

Type Casting (.int, etc.): Converts a tensor from one data type to another (e.g., from float to integer)."""
print("DATA:", data)
print("DATA TYPE:", data.dtype)
print("-" * 45)

# Cast the tensor to a int type
int_tensor = data.int()

print("\nCASTED DATA:", int_tensor)
print("CASTED DATA TYPE", int_tensor.dtype)