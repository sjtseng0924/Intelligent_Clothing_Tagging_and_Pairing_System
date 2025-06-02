import os
import subprocess
from collections import Counter
import shutil

# Directory structure
base_dir = "../../dataset/Test-Dataset-Category"

# Total matches counter across all sets
total_matches_all = 0

# Create filter directories inside Test-Dataset-Category
filter_top_dir = os.path.join(base_dir, "filter_top")
filter_bottom_dir = os.path.join(base_dir, "filter_bottom")
os.makedirs(filter_top_dir, exist_ok=True)
os.makedirs(filter_bottom_dir, exist_ok=True)

# Process each dataset pair (1-5)
for num in range(1, 6):
    top_dir = f"{base_dir}/top-20-{num}"
    bottom_dir = f"{base_dir}/bottom-20-{num}"
    
    print(f"\n==== Processing dataset {num} ====")
    print(f"Top directory: {top_dir}")
    print(f"Bottom directory: {bottom_dir}")
    
    positive_counter = Counter()
    total_matches = 0
    
    # Skip if directory doesn't exist
    if not os.path.exists(top_dir) or not os.path.exists(bottom_dir):
        print(f"Directory not found for dataset {num}, skipping.")
        continue
    
    for top_fname in os.listdir(top_dir):
        if not top_fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        top_path = os.path.join(top_dir, top_fname)
        
        # Extract the top ID from filename (assuming format like XXXX_top.jpg)
        top_id = top_fname.split("_")[0]
        
        # Call inference.py and get positive results
        result = subprocess.check_output([
            "python3", "inference.py",
            "--top", top_path,
            "--bottom_dir", bottom_dir,
            "--top_k", "20"  # Ensure all are listed
        ], universal_newlines=True)
        
        # Parse positive results
        for i, line in enumerate(result.splitlines()):
            if line.startswith("Positive Result"):
                # e.g., Positive Result - 1234567_bottom.jpg: 0.8765
                parts = line.split("-")
                if len(parts) >= 2:
                    fname_part = parts[1].split(":")[0].strip()
                    # Extract ID from bottom filename
                    bottom_id = fname_part.split("_")[0]
                    positive_counter[bottom_id] += 1
                    
                    # Check if top and bottom IDs match
                    if top_id == bottom_id:
                        print(f"Match found: {top_fname} matches with {fname_part} (Position: {i+1})")
                        total_matches += 1
                        if i <= 10:
                            # Copy the matching files
                            top_dest = os.path.join(filter_top_dir, top_fname)
                            bottom_dest = os.path.join(filter_bottom_dir, fname_part)
                            shutil.copy2(top_path, top_dest)
                            shutil.copy2(os.path.join(bottom_dir, fname_part), bottom_dest)
                            print(f"Copied matching pair to filter folders: {top_fname} and {fname_part}")
    
    print(f"\nDataset {num} matches found: {total_matches}")
    total_matches_all += total_matches

print("\n==== Summary ====")
print(f"Total matches found across all datasets: {total_matches_all}")
