import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Sample data: Simulate time series data for different locations
locations = ["Location A", "Location B", "Location C"]
time_points = pd.date_range(start="2022-01-01", periods=100, freq="D")
data = {loc: np.random.randn(len(time_points)) for loc in locations}

# Create a PDF file to store the plots
pdf_filename = "report.pdf"
with PdfPages(pdf_filename) as pdf:
    # Loop through each location to create and save plots
    for loc in locations:
        plt.figure(figsize=(8, 4))  # Set the figure size
        plt.plot(time_points, data[loc], marker='o', linestyle='-', label=loc)
        plt.title(f"Time Series Data for {loc}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Save the current plot into the PDF file
        pdf.savefig()
        plt.close()  # Close the figure to free memory

    # Optional: Add a summary page or additional text
    plt.figure()
    plt.text(0.5, 0.5, "Summary of Report", fontsize=20, ha='center')
    pdf.savefig()
    plt.close()

print(f"Report saved as {pdf_filename}.")
