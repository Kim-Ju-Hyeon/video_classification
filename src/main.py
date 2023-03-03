import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add all subdirectories to the Python path
for subdir in os.listdir(current_dir):
    subdir_path = os.path.join(current_dir, subdir)
    if os.path.isdir(subdir_path):
        sys.path.append(subdir_path)