import csv
import random

# Create a CSV file with 10,000 rows
def generate_csv(filename):
    headers = ['id', 'value']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for i in range(1, 10001):
            writer.writerow([i, random.randint(1, 100)])

generate_csv('data/small_10k.csv')

