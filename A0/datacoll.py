import os
import subprocess
import csv
import numpy as np
import time

# Paths
INPUT_PATH = "./input_path/"
OUTPUT_PATH = "./output_path/"
DATA_FILE = "data.csv"

# Matrix sizes and types for testing
MATRIX_SIZES = [1000, 2000, 3000, 4000, 5000]
LOOP_TYPES = [0, 1, 2, 3, 4, 5]

# Function to run perf and collect data
def run_perf(command):
    perf_command = f"perf stat -e cache-references,cache-misses,cycles,instructions {command}"
    process = subprocess.Popen(perf_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    _, stderr = process.communicate()
    process.wait()

    # Extract metrics from perf output
    metrics = {
        "cache-references": None,
        "cache-misses": None,
        "cycles": None,
        "instructions": None
    }
    for line in stderr.splitlines():
        for metric in metrics:
            if metric in line:
                metrics[metric] = int(line.split()[0].replace(',', ''))

    return metrics

# Function to execute a single test case
def execute_test_case(type, rows, cols, inner):
    # Generate random matrices
    mtx_A = np.random.random(size=(rows, inner)) * 1e2
    mtx_B = np.random.random(size=(inner, cols)) * 1e2

    # Create directories if necessary
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save matrices to binary files
    with open(f"{INPUT_PATH}/mtx_A.bin", "wb") as f:
        f.write(mtx_A.tobytes())
    with open(f"{INPUT_PATH}/mtx_B.bin", "wb") as f:
        f.write(mtx_B.tobytes())

    # Compile the code
    os.system("make")

    # Run the matrix multiplication executable with perf
    command = f"./main {type} {rows} {inner} {cols} {INPUT_PATH} {OUTPUT_PATH}"
    start_time = time.perf_counter()
    metrics = run_perf(command)
    duration = time.perf_counter() - start_time

    # Validate results
    with open(f"{OUTPUT_PATH}/mtx_C.bin", "rb") as f:
        student_result = np.frombuffer(f.read(), dtype=mtx_A.dtype).reshape(rows, cols)
    expected_result = np.dot(mtx_A, mtx_B)

    is_correct = np.allclose(student_result, expected_result, rtol=1e-10, atol=1e-12)

    return is_correct, duration, metrics

# Main function to execute tests and store data
def main():
    with open(DATA_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Type", "Matrix Size", "Time (s)", "Cache References", "Cache Misses", "Cycles", "Instructions", "Accuracy"])

        for size in MATRIX_SIZES:
            rows = cols = size
            inner = size

            for loop_type in LOOP_TYPES:
                print(f"Running test for type {loop_type} with size {size}x{size}...")

                is_correct, duration, metrics = execute_test_case(loop_type, rows, cols, inner)

                writer.writerow([
                    loop_type,
                    f"{size}x{size}",
                    duration,
                    metrics["cache-references"],
                    metrics["cache-misses"],
                    metrics["cycles"],
                    metrics["instructions"],
                    "Pass" if is_correct else "Fail"
                ])

                print(f"Test {'passed' if is_correct else 'failed'} in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
