import tkinter as tk

class TextRedirector:
    def __init__(self, text_widget):
        self.output = text_widget

    def cprint(self, string, color='white'):
        self.output.insert(tk.END, string + '\n', color)
        self.output.see(tk.END)

def demo_function():
    stdout.cprint("Hello, world!", 'green')
    stdout.cprint("This is an error message.", 'red')
    stdout.cprint("This is just informational.", 'blue')
    stdout.cprint("Warning: Be cautious!", 'yellow')
    # Printing more lines to demonstrate scrolling
    for i in range(100):
        stdout.cprint(f"Line {i+1}", 'green' if i % 4 == 0 else 'red' if i % 4 == 1 else 'blue' if i % 4 == 2 else 'yellow')

# Create the main window
window = tk.Tk()
window.title("Output Redirection with Colored Text")

# Create a text widget for output and a scrollbar
text = tk.Text(window, height=10, width=50, bg='black')
scrollbar = tk.Scrollbar(window, command=text.yview)
text.configure(yscrollcommand=scrollbar.set)

# Layout the text widget and scrollbar
text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Define text styles
text.tag_configure('white', foreground='white')
text.tag_configure('red', foreground='red')
text.tag_configure('green', foreground='green')
text.tag_configure('blue', foreground='blue')
text.tag_configure('yellow', foreground='yellow')

# Instantiate redirector
stdout = TextRedirector(text)

# Button to trigger a function that prints
button = tk.Button(window, text="Run Function", command=demo_function)
button.pack()

# Start the Tkinter event loop
window.mainloop()
