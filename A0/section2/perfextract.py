import re
import csv



def addcontents(filename='perf_report_1000_1000_0.txt',perm= 0,size=1000):
# Replace 'your_file.txt' with the path to your text file
    file_path = filename

    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expression to match the pattern
    pattern1 = r"# Event count \(approx.\): (\d+)"
    count = 0
    pattern2 = r"(\d+\.\d+%)"
    
    numbers = []
    divisor = 10**9
    percent = 0


    # Extract the number
    for line in lines:
        match1 = re.search(pattern1, line)
        
        if match1:
#            print("Event count:", match1.group(1))
            numbers.append(int(match1.group(1)))
        if count < 3:
            matches = re.findall(pattern2, line)
            for match in matches:
                count += 1
                if count == 3:
#                    print("Percentage:", match)
                    percent = float(match[:-1])
                    break
    
    user_time = numbers[0]/divisor
    cycles = numbers[1]
    instructions = numbers[2]
    cache_ref = numbers[3]
    cache_misses = numbers[4]
    
    user_time_matrix = (percent/100)*user_time
    miss_rate = (cache_misses/cache_ref)*100
    hit_rate = 100 - miss_rate
    instructions_per_cycle = instructions/cycles

    # print(cache_misses)
    
    row = [perm, size, user_time, user_time_matrix, percent, cache_misses, hit_rate, instructions_per_cycle]
    
    return row
    
    
sizes = [1000,2000,3000,4000,5000]

with open('data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    for size in sizes:
        for type in range(6):
            filename = 'perf_report' + ('_' + str(size))*2 + '_' + str(type) + '.txt'
            #print(filename)
            row = addcontents(filename,type,size)
            writer.writerow(row)

    # filename = 'perf_report' + ('_' + str(1000))*2 + '_' + str(2) + '.txt'
    # #print(filename)
    # row = addcontents(filename,2,1000)
    # writer.writerow(row)





