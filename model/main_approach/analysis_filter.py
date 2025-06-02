import os
import subprocess
from collections import Counter

# os base dir = ".."
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.sys.path.append(project_root)

# Directory structure
base_dir = os.path.join(project_root, "dataset/Test-Dataset-Category")

# Process each dataset pair (1-5)
top_dir = f"{base_dir}/filter_top"
bottom_dir = f"{base_dir}/filter_bottom"

positive_counter = Counter()
total_matches = 0

for top_fname in os.listdir(top_dir):
    if not top_fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    top_path = os.path.join(top_dir, top_fname)
    
    # Extract the top ID from filename (assuming format like XXXX_top.jpg)
    top_id = top_fname.split("_")[0]
    
    # Call inference.py and get positive results
    inference_script = os.path.join(os.path.dirname(__file__), "inference.py")
    result = subprocess.check_output([
        "python3", inference_script,
        "--top", top_path,
        "--bottom_dir", bottom_dir,
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
                
                output_txt = os.path.join(project_root, "model/main_approach", "top_bottom_rank.txt")
                with open(output_txt, "a") as f:
                    # Check if top and bottom IDs match
                    if top_id == bottom_id:
                        f.write(f"{top_id} {i+1}\n")
                        print(f"Match found: {top_fname} matches with {fname_part} (Position: {i+1})")
                        total_matches += 1

print(f"\nDataset matches found: {total_matches}")

