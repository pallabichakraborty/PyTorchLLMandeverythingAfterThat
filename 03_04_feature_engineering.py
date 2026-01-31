"""Exercise 4: Feature Engineering for Taxi Fares
You are working with a dataset of taxi trips. You have a tensor, trip_data, where each row is a trip and the columns represent [distance (km), hour_of_day (24h)].

Your goal is to engineer a new binary feature called is_rush_hour_long_trip. This feature should be True (or 1) only if a trip meets both of the following criteria:

It's a long trip (distance > 10 km).
It occurs during a rush hour (8-10 AM or 5-7 PM, i.e., [8, 10) or [17, 19)).
To achieve this, you will need to:

Slice the trip_data tensor to isolate the distance and hour columns.
Use logical and comparison operators to create boolean masks for each condition (long trip, morning rush, evening rush).
Combine these masks to create the final is_rush_hour_long_trip feature.
Reshape this new 1D feature tensor into a 2D column vector and convert its data type to float so it can be combined with the original data.
"""
import torch
import numpy as np
import pandas as pd

trip_data = torch.tensor([
    [5.3, 7],   # Not rush hour, not long
    [12.1, 9],  # Morning rush, long trip -> RUSH HOUR LONG
    [15.5, 13], # Not rush hour, long trip
    [6.7, 18],  # Evening rush, not long
    [2.4, 20],  # Not rush hour, not long
    [11.8, 17], # Evening rush, long trip -> RUSH HOUR LONG
    [9.0, 9],   # Morning rush, not long
    [14.2, 8]   # Morning rush, long trip -> RUSH HOUR LONG
], dtype=torch.float32)


print("ORIGINAL TRIP DATA (Distance, Hour):\n\n", trip_data)
print("-" * 55)

# 1. Slice the main tensor to get 1D tensors for each feature.
distances = None
hours = None

# 2. Create boolean masks for each condition.
is_long_trip = None
is_morning_rush = None
is_evening_rush = None

# 3. Combine masks to identify rush hour long trips.
# A trip is a rush hour long trip if it's (a morning OR evening rush) AND a long trip.
is_rush_hour_long_trip_mask = None

# 4. Reshape the new feature into a column vector and cast to float.
new_feature_col = None

print("\n'IS RUSH HOUR LONG TRIP' MASK: ", is_rush_hour_long_trip_mask)
print("\nNEW FEATURE COLUMN (Reshaped):\n\n", new_feature_col)

# You can now concatenate this new feature to the original data
enhanced_trip_data = torch.cat((trip_data, new_feature_col), dim=1)
print("\nENHANCED DATA (with new feature at the end):\n\n", enhanced_trip_data)