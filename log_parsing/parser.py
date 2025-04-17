import sys
import os
import csv

def extract_csv_from_log(filepath):
    csv_data = []
    recording = False

    with open(filepath, 'r') as file:
        for line in file:
            stripped = line.strip()

            if recording:
                if not stripped:  # Stop at blank line
                    break
                csv_data.append(stripped.split(';'))
            elif stripped == "Activities log:":
                recording = True

    return csv_data

def save_to_csv(records, output_filename):
    headers = records[0]
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records[1:])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = base_name + ".csv"

    extracted_data = extract_csv_from_log(input_path)
    if extracted_data:
        save_to_csv(extracted_data, output_path)
        print(f"Saved extracted CSV to {output_path}")
    else:
        print("No CSV data found after 'Activities log:' marker.")
