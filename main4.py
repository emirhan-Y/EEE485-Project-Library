import tkinter as tk
from PIL import Image, ImageTk
import os


def load_image_on_canvas():
    # Load an image file
    _path = os.path.abspath('_data/final/gokce/gokce_4.jpg')
    image = Image.open(_path)
    photo = ImageTk.PhotoImage(image)

    # Place the image in the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo  # keep a reference!

    # Create an entry widget and place it over the image
    label = tk.Label(canvas, text="Your Text Here", bg='white', fg='black')
    canvas.create_window(200, 50, window=label, width=200)  # Adjust position and size as needed


# Create the main window
window = tk.Tk()
window.title("Entry Over Image")

# Create a canvas widget
canvas = tk.Canvas(window, width=400, height=400)
canvas.pack()

# Button to load image and create entry field
button = tk.Button(window, text="Load Image and Add Entry", command=load_image_on_canvas)
button.pack()

# Start the Tkinter event loop
window.mainloop()
