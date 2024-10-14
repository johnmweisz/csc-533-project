import matplotlib.pyplot as plt
import pandas as pd

def generate_histogram_from_csv(csv_file):
    # Load the data from the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Set classification as the index
    df.set_index('classification', inplace=True)
    
    # Plot the bar chart
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Histogram of Accurate Classifications for Noise Levels')
    plt.xlabel('Classification')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.legend(title='Noise Levels and Target')
    plt.tight_layout()
    
    # Show the plot
    plt.show()

generate_histogram_from_csv('results.csv')
