import tkinter as tk


class Console(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter Console")

        self._console = tk.Text(self, height=15, width=80, bg='black', fg='white', insertbackground='white')
        self._console.pack(padx=10, pady=10)

        self._attach_binds()

        # Start the console with a prompt
        self._insert_text(">>> ")

    def _attach_binds(self):
        self._console.bind("<Return>", self._on_enter_pressed)
        self._console.bind("<Key>", self._on_key_press)
        self._console.bind("<Button-1>", self._on_mouse_click)

    def _insert_text(self, text):
        self._console.insert(tk.END, text)
        self._console.mark_set(tk.INSERT, tk.END)
        self._console.see(tk.END)

    def _on_enter_pressed(self, event):
        # Prevent the default return key behavior
        event.widget.mark_set(tk.INSERT, "end-1c linestart+4c")
        line = event.widget.get("insert linestart", "insert lineend")
        self._insert_text("\n")
        self.handle_command(line.strip()[4:])  # skip prompt
        self._insert_text(">>> ")
        return "break"  # prevent further processing

    def _on_key_press(self, event):
        # Get current insertion cursor index
        insert_index = self._console.index(tk.INSERT)
        input_start_index = self._console.index("end-1c linestart+4c")

        # Disable modification if the cursor is outside the allowable area
        if self._console.compare(insert_index, "<", input_start_index):
            return "break"

        # Handling for backspace
        if event.keysym == "BackSpace":
            if self._console.compare(insert_index, "==", input_start_index):
                return "break"  # Prevent deletion of prompt

        # Allow normal processing within the input line
        return None

    def _on_mouse_click(self, event):
        # Restrict mouse click setting cursor position only to the last line
        self._console.mark_set(tk.INSERT, "end-1c linestart+4c")
        self._console.see(tk.END)
        return "break"  # Prevents default handler

    def handle_command(self, command):
        # You can process commands here
        self._insert_text(f"Command entered: {command}\n")

    def cprint(self, string, color='white', eol=False):
        if eol:
            self._console.insert(tk.END, string + '\n', color)
        else:
            self._console.insert(tk.END, string + ' ', color)
        self._console.mark_set(tk.INSERT, tk.END)
        self._console.see(tk.END)


if __name__ == "__main__":
    app = Console()
    app.mainloop()