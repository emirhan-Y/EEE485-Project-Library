import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageGrab
import cv2
import numpy as np


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Canvas")

        # Layout: divide the window into drawing canvas and control panel
        self.control_frame = tk.Frame(root, padx=5, pady=5)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(root, padx=5, pady=5)
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Create the canvas
        self.canvas = tk.Canvas(self.canvas_frame, width=126 * 5 + 3, height=50 * 5 + 3, bg='white')
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Bind mouse events to methods
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_paint)  # Bind left mouse button press to start a new line
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)  # Bind left mouse button release to reset position
        self.canvas.bind("<Button-3>", self.reset_canvas)

        # Initialize last position
        self.last_x, self.last_y = None, None

        # Control Panel Elements
        self.clear_button = tk.Button(self.control_frame, text="Clear", command=self.reset_canvas)
        self.clear_button.pack(fill=tk.X)

        self.color_button = tk.Button(self.control_frame, text="Change Color", command=self.change_color)
        self.color_button.pack(fill=tk.X)

        # Add Save Button
        self.save_btn = tk.Button(self.control_frame, text="Save Drawing", command=self.save_drawing)
        self.save_btn.pack(fill=tk.X)

        # Set initial drawing color
        self.color = 'black'

    def start_paint(self, event):
        # Set the starting point for the new line
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=5, fill='black',
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
        # Update the last position to the current position
        self.last_x, self.last_y = event.x, event.y

    def reset_position(self, event):
        # Reset the last position to prevent drawing a line from the last point when starting a new one
        self.last_x, self.last_y = None, None

    def reset_canvas(self, event=None):  # default None for event to allow button use
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None

    def change_color(self):
        # This function will change the drawing color
        # You could enhance this to open a color picker dialog
        self.color = 'blue' if self.color == 'black' else 'black'

    def save_drawing(self):
        # Ensure the entire window layout is finalized
        self.root.update()  # This ensures the layout is complete

        # Calculate the absolute coordinates of the canvas
        x = self.root.winfo_x() + self.canvas_frame.winfo_x() + self.canvas.winfo_x() + 10
        y = self.root.winfo_y() + self.canvas_frame.winfo_y() + self.canvas.winfo_y() + 33
        x1 = x + self.canvas.winfo_width() - 4
        y1 = y + self.canvas.winfo_height() - 4

        # Grab the image, including only the canvas
        image = ImageGrab.grab().crop((x,y,x1,y1))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        data_instance = cv2.resize(cv_image, (126, 50), interpolation=cv2.INTER_AREA)

        # Save the image using a file dialog
        filepath = filedialog.asksaveasfilename(defaultextension='.png',
                                                filetypes=[("PNG files", '*.png'), ("JPEG files", '*.jpeg')])
        if filepath:
            cv2.imwrite(filepath, data_instance)


# Create the main window and pass it to the DrawingApp
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
