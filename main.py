import tkinter as tk
from app import drawing_app

if __name__ == '__main__':
    # Create the main window and pass it to the DrawingApp
    root = tk.Tk()
    app = drawing_app(root)
    root.mainloop()
