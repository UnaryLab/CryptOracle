import os

def parse_data(file_path):
    time_values = []
    energy_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            
            # Skip lines that do not have at least two parts or contain headers
            if len(parts) >= 2:
                try:
                    # Try to parse the first two parts as floats (time and energy values)
                    time_value = float(parts[0])  # First part is time
                    energy_value = float(parts[1])  # Second part is energy
                    
                    # Collect the parsed values
                    time_values.append(time_value)
                    energy_values.append(energy_value)
                except ValueError:
                    # If conversion fails, skip the line (likely a header or invalid line)
                    continue
    
    return time_values, energy_values

# File path to the data file
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../out/temp/perf_out.perf")
time_values, energy_values = parse_data(file_path)

# Calculate averages of the parsed values
average_time = time_values[-1] if time_values else 0  # Using the last time value. Not average time, but leaving the variable name for parsing purposes
total_energy = sum(energy_values) if energy_values else 0  # Summing the energy values

# Output the results
print(f"\nAverage Time: {average_time:.6f} seconds")
print(f"Total Energy Usage: {total_energy:.2f} Joules\n")