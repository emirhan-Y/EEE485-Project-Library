import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_data(x, y):
    plot.clear()  # Clear the current plot
    plot.plot(x, y, marker='o')  # Create a new plot with the given x and y data
    canvas.draw()  # Redraw the canvas

# Create the main window
root = tk.Tk()
root.title("Dynamic Plotting in Tkinter")

# Create a figure to contain the plot
fig = Figure(figsize=(5, 4), dpi=100)
plot = fig.add_subplot(111)

# Create a canvas as a widget in Tkinter and add the figure to it
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Define data sets
data1 = ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
data2 = ([1, 2, 3, 4, 5], [6, 5, 4, 3, 2])
data3 = ([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])

# Buttons to switch between plots
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

button1 = tk.Button(button_frame, text="Plot 1", command=lambda: plot_data(*data1))
button1.pack(side=tk.LEFT)

button2 = tk.Button(button_frame, text="Plot 2", command=lambda: plot_data(*data2))
button2.pack(side=tk.LEFT)

button3 = tk.Button(button_frame, text="Plot 3", command=lambda: plot_data(*data3))
button3.pack(side=tk.LEFT)

# Start the Tkinter main loop
root.mainloop()
