import numpy as np, os, time

def execute_test_case(type, number_row1, number_col1, number_col2, path_input, path_output):
    # Generate random matrices
    mtx_A = np.random.random(size = (number_row1, number_col1)) * 1e2 # dtype = float64
    mtx_B = np.random.random(size = (number_col1, number_col2)) * 1e2 # dtype = float64

    # Matrix multiplication
    mtx_C = (mtx_A @ mtx_B).flatten() # dtpye = float64

    # Create directories if does not exist
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Store the input matrices
    with open(f"{path_input}/mtx_A.bin", "wb") as fp:
        fp.write(mtx_A.tobytes())
    with open(f"{path_input}/mtx_B.bin", "wb") as fp:
        fp.write(mtx_B.tobytes())
    
    # Compile the students code
    os.system("make")

    # Execute the students code
    time_start = time.perf_counter()
    os.system(f"perf record -e task-clock,cycles,instructions,cache-references,cache-misses -F 1000 -g ./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}")
    
    # Create the directory for perf reports if it doesn't exist
    perf_report_dir = "section2"
    if not os.path.exists(perf_report_dir):
        os.makedirs(perf_report_dir)

    # Generate the output file path for perf report
    outputfile = f"{perf_report_dir}/perf_report_{number_row1}_{number_col2}_{type}.txt"
    os.system(f"perf report --stdio > {outputfile}")
    
    time_duration = time.perf_counter() - time_start # in seconds

    # Get the students output matrix
    with open(f"{path_output}/mtx_C.bin", "rb") as fp:
        student_result = np.frombuffer(fp.read(), dtype=mtx_C.dtype)
    
    # Check if the result matrix dimensions match
    if mtx_C.shape != student_result.shape:
        print("The result matrix shape didn't match")
        return False, time_duration
    
    # Check if the student's result is close to the numpy's result within a tolerance
    result = np.allclose(mtx_C, student_result, rtol=1e-10, atol=1e-12)

    return result, time_duration

if __name__ == "__main__":
    # 0 = IJK, 1 = IKJ, 2=JIK, 3=JKI, 4=KIJ, 5=KJI
    MATRIX_SIZES = [1000, 2000, 3000, 4000, 5000]
    LOOP_TYPES = [0, 1, 2, 3, 4, 5]
    for size in MATRIX_SIZES:
        rows = cols = size
        inner = size
        for loop_type in LOOP_TYPES:
            print(f"Running test for type {loop_type} with size {size}x{size}...")
            result, time_duration = execute_test_case(loop_type, rows, cols, inner, "./input_path/", "./output_path/")
            if result:
                print("Test Case passed")
            else:
                print("Test Case failed")
            print(f"Time taken: {time_duration} seconds")
