import tkinter as tk
from PIL import Image, ImageTk
import os


def load_actual_image():
    # Load an image file
    _path = os.path.abspath('_data/final/gokce/gokce_4.jpg')
    actual_image = Image.open(_path)
    upscaled_image = actual_image.resize((630, 250), Image.BICUBIC)
    actual_photo = ImageTk.PhotoImage(upscaled_image)

    # Place the image in the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=actual_photo)
    canvas.image = actual_photo  # keep a reference!

    # Create an entry widget and place it over the image
    label = tk.Label(canvas, text="gokce_4.jpg", bg='white', fg='black')
    canvas.create_window(315, 20, window=label, width=200)  # Adjust position and size as needed

# Create the main window
window = tk.Tk()
window.title("Entry Over Image")

# Create a placeholder image (200x200 white image)
placeholder = Image.new('RGB', (630, 250), 'white')
placeholder_photo = ImageTk.PhotoImage(placeholder)

# Create a canvas widget
canvas = tk.Canvas(window, width=630, height=250)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=placeholder_photo)
canvas.image = placeholder_photo  # keep a reference!

# Button to load image and create entry field
button = tk.Button(window, text="Load Image and Add Entry", command=load_actual_image)
button.pack()

# Start the Tkinter event loop
window.mainloop()
