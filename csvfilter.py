import csv

def filter_and_transform_csv(input_file, output_file):
    # List of values to filter out from the first cell
    filter_values = {'1', '5', '6'}
    
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            if row and row[0].strip() not in filter_values:
                # Check and transform values in the first cell
                if row[0].strip() == '2':
                    row[0] = '1'
                elif row[0].strip() == '3':
                    row[0] = '2'
                
                # Write the transformed row to the output CSV file
                writer.writerow(row)

def main():
    input_file = 'varun_pics_augmenteddos.csv'
    output_file = 'varun_nodisgust.csv'
    
    filter_and_transform_csv(input_file, output_file)
    print("Filtered and transformed CSV created successfully.")

if __name__ == "__main__":
    main()
