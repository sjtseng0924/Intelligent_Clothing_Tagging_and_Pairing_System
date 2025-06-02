input_path = "rank_main.txt"
output_path = "rank_main.txt"  # overwrite the original file

with open(input_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Sort by the first column (as integer)
lines.sort(key=lambda x: int(x.split()[0]))

with open(output_path, "w") as f:
    for line in lines:
        f.write(line + "\n")

print("Sorted and saved to", output_path)