"""Exercise 1: Analyzing Monthly Sales Data
You're a data analyst at an e-commerce company. You've been given a tensor representing the monthly sales of three different products over a period of four months. Your task is to extract meaningful insights from this data.

The tensor sales_data is structured as follows:

Rows represent the products (Product A, Product B, Product C).

Columns represent the months (Jan, Feb, Mar, Apr).

Your goals are:

Calculate the total sales for Product B (the second row).
Identify which months had sales greater than 130 for Product C (the third row) using boolean masking.
Extract the sales data for all products for the months of Feb and Mar (the middle two columns).
"""

import  torch
import numpy as np
import pandas as pd

sales_data = torch.tensor([[100, 120, 130, 110],   # Product A
                           [ 90,  95, 105, 125],   # Product B
                           [140, 115, 120, 150]    # Product C
                          ], dtype=torch.float32)

print("ORIGINAL SALES DATA:\n\n", sales_data)
print("-" * 45)

### START CODE HERE ###

# 1. Calculate total sales for Product B.
total_sales_product_b = sales_data[1].sum()

# 2. Find months where sales for Product C were > 130.
high_sales_mask_product_c = sales_data[2] > 130

# 3. Get sales for Feb and Mar for all products.
sales_feb_mar = sales_data[:, 1:3]

### END CODE HERE ###

print("\nTotal Sales for Product B:                   ", total_sales_product_b)
print("\nMonths with >130 Sales for Product C (Mask): ", high_sales_mask_product_c)
print("\nSales for Feb & Mar:\n\n", sales_feb_mar)