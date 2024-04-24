import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageGrab, ImageTk
import cv2
import numpy as np

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class drawing_app:
    def __init__(self, root):
        self.drawing_thickness = 5  # default drawing thickness

        self._root = root
        self._root.title("Signature Canvas")

        self._font = ("Consolas", 14, "bold")

        self._generate_layout()
        self._populate_layout()

        self._last_x, self._last_y = None, None
        # self.attach_bind()

        # Set initial drawing color
        self._color = 'black'

    def _generate_layout(self):
        """
        Creates the window frame hierarchy
        """
        # The screen split into three parts: left middle and right.
        # Now we further split left part into its parts: canvas, data_point and plot areas
        # left frame
        self._left_frame = tk.Frame(self._root, padx=5, pady=5, highlightbackground="black", highlightcolor="black",
                                    highlightthickness=1)
        self._left_frame.pack(side=tk.LEFT, expand=False, fill=tk.BOTH)
        # left frame -> canvas frame
        self._canvas_frame = tk.Frame(self._left_frame, padx=5, pady=5, highlightbackground="black",
                                      highlightcolor="black", highlightthickness=1)
        self._canvas_frame.pack(side=tk.TOP, expand=False, fill=tk.BOTH)
        # left frame -> canvas frame -> drawable settings frame
        self._drawable_settings_frame = tk.Frame(self._canvas_frame, highlightbackground="black",
                                                 highlightcolor="black", highlightthickness=1)
        self._drawable_settings_frame.pack(side=tk.BOTTOM, expand=True, fill=tk.X)
        self._drawable_settings_frame.columnconfigure((0, 1, 2), weight=1)
        # left frame -> data point frame
        self._data_point_frame = tk.Frame(self._left_frame, padx=5, pady=5, highlightbackground="black",
                                          highlightcolor="black", highlightthickness=1)
        self._data_point_frame.pack(side=tk.TOP, expand=False, fill=tk.BOTH)
        # left frame -> data point frame -> data point settings frame
        self._data_point_settings_frame = tk.Frame(self._data_point_frame, padx=5, pady=5, highlightbackground="black",
                                                   highlightcolor="black", highlightthickness=1)
        self._data_point_settings_frame.pack(side=tk.BOTTOM, expand=True, fill=tk.X)
        self._data_point_settings_frame.columnconfigure((0, 1, 2), weight=1)
        # left frame -> console frame
        self._console_frame = tk.Frame(self._left_frame, padx=5, pady=5, highlightbackground="black",
                                       highlightcolor="black", highlightthickness=1)
        self._console_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        # middle-right holder
        self._middle_right_frame = tk.Frame(self._root, padx=5, pady=5, highlightbackground="black",
                                            highlightcolor="black", highlightthickness=1)
        self._middle_right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        # middle frame
        self._middle_frame = tk.Frame(self._middle_right_frame, padx=5, pady=5, highlightbackground="black",
                                      highlightcolor="black", highlightthickness=1)
        self._middle_frame.pack(side=tk.LEFT, expand=False, fill=tk.Y)
        # right frame
        self._right_frame = tk.Frame(self._middle_right_frame, padx=5, pady=5, highlightbackground="black",
                                     highlightcolor="black", highlightthickness=1)
        self._right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        # middle frame -> knn control frame
        self._knn_control_frame = tk.Frame(self._middle_frame, highlightbackground="black", highlightcolor="black",
                                           highlightthickness=1)
        self._knn_control_frame.grid(row=0, column=0, sticky='news')
        # middle frame -> logr control frame
        self._logr_control_frame = tk.Frame(self._middle_frame, highlightbackground="black", highlightcolor="black",
                                            highlightthickness=1)
        self._logr_control_frame.grid(row=1, column=0, sticky='news')
        # middle frame -> cnn control frame
        self._cnn_control_frame = tk.Frame(self._middle_frame, highlightbackground="black", highlightcolor="black",
                                           highlightthickness=1)
        self._cnn_control_frame.grid(row=2, column=0, sticky='news')
        self._middle_frame.rowconfigure((0, 1, 2), weight=1)

    def _populate_layout(self):
        """
        Adds necessary elements to their respective frames
        """
        # Populate each frame with its required components.
        # left frame
        pass
        # left frame -> canvas frame
        self._canvas = tk.Canvas(self._canvas_frame, width=126 * 5, height=50 * 5, bg='white')
        self._canvas.pack(side=tk.TOP, expand=False, fill=tk.BOTH)
        # left frame -> canvas frame -> drawable settings frame
        self._clear_button = tk.Button(self._drawable_settings_frame, text="Clear", command=self._reset_canvas,
                                       font=self._font)
        self._clear_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self._thickness_inc_btn = tk.Button(self._drawable_settings_frame, text="+Thickness",
                                            command=self._thickness_increase, font=self._font)
        self._thickness_inc_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self._thickness_dec_btn = tk.Button(self._drawable_settings_frame, text="-Thickness",
                                            command=self._thickness_decrease, font=self._font)
        self._thickness_dec_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self._push_drawing_knn_btn = tk.Button(self._drawable_settings_frame, text="Push KNN",
                                               command=self.push_drawing_knn, font=self._font)
        self._push_drawing_knn_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self._push_drawing_logr_btn = tk.Button(self._drawable_settings_frame, text="Push LogR",
                                                command=self.push_drawing_logr, font=self._font)
        self._push_drawing_logr_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self._push_drawing_cnn_btn = tk.Button(self._drawable_settings_frame, text="Push NN",
                                               command=self.push_drawing_nn,
                                               font=self._font)
        self._push_drawing_cnn_btn.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        # left frame -> data point frame
        placeholder = Image.new('RGB', (630, 250), 'white')  # Create a placeholder image (200x200 white image)
        placeholder_photo = ImageTk.PhotoImage(placeholder)
        self._data_point_canvas = tk.Canvas(self._data_point_frame, width=630, height=250)
        self._data_point_canvas.pack(side=tk.TOP, expand=False, fill=tk.BOTH)
        self._data_point_canvas.create_image(0, 0, anchor=tk.NW, image=placeholder_photo)
        self._data_point_canvas.image = placeholder_photo  # keep a reference!
        # left frame -> data point frame -> data point settings frame
        self._load_data_btn = tk.Button(self._data_point_settings_frame, text="Load Data",
                                        command=self._load_data_cmd, font=self._font)
        self._load_data_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self._random_test_btn = tk.Button(self._data_point_settings_frame, text="Random Test",
                                          command=self._random_test_cmd, font=self._font)
        self._random_test_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self._train_all_btn = tk.Button(self._data_point_settings_frame, text="placeholder3",
                                        command=self._train_all_cmd, font=self._font)
        self._train_all_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        # left frame -> console frame
        self._console = tk.Text(self._console_frame, height=10, width=50, bg='black')
        self._console_scrollbar = tk.Scrollbar(self._console_frame, command=self._console.yview)
        self._console.configure(yscrollcommand=self._console_scrollbar.set)
        self._console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._console.tag_configure('white', foreground='white')
        self._console.tag_configure('red', foreground='red')
        self._console.tag_configure('green', foreground='green')
        self._console.tag_configure('blue', foreground='blue')
        self._console.tag_configure('yellow', foreground='yellow')
        self.stdout = redirect_print(self._console)
        self.stdout.cprint('*Signature Recognition App Console v0.6*', color='yellow')
        self.stdout.cprint('Note that this is not a python console, it is here for basic input output')
        self.stdout.cprint('functionality')
        # middle frame -> knn control frame
        self._knn_control_frame_label = tk.Label(self._knn_control_frame, text="K Nearest Neighbors Controls",
                                                 font=self._font)
        self._knn_control_frame_label.pack()
        # middle frame -> logr control frame
        self._logr_control_frame_label = tk.Label(self._logr_control_frame, text="Logistic Regression Controls",
                                                  font=self._font)
        self._logr_control_frame_label.pack()
        # middle frame -> cnn control frame
        self._cnn_control_frame_label = tk.Label(self._cnn_control_frame, text="Convolutional Neural Network Controls",
                                                 font=self._font)
        self._cnn_control_frame_label.pack()
        # right frame
        self._main_plot_fig = Figure(figsize=(3, 3), dpi=100)  # Create a figure and add a subplot
        self._main_plot_fig.suptitle('PLACE_HOLDER_TITLE')
        self._main_plot_fig.supxlabel('PLACE_HOLDER_X_LABEL')
        self._main_plot_fig.supylabel('PLACE_HOLDER_Y_LABEL')
        self._main_plot_fig_ax = self._main_plot_fig.add_subplot(111)
        self._main_plot_fig_ax.plot([], [])  # Initially display an empty plot
        self._main_plot = FigureCanvasTkAgg(self._main_plot_fig, master=self._right_frame)
        self._main_plot.draw()
        self._main_plot.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH)

    def _load_data_cmd(self):
        pass

    def _random_test_cmd(self):
        pass

    def _train_all_cmd(self):
        pass

    def _knn_dim_figure_plot(self):
        # Data for plotting
        t = range(0, 3)
        s = [i ** 2 for i in t]

        # Clear the previous plot and create a new plot
        self._main_plot_fig_ax.clear()
        self._main_plot_fig_ax.plot(t, s)
        self._main_plot.draw()

    def _placeholder1_btn1_cmd(self):
        pass

    def _placeholder1_btn2_cmd(self):
        pass

    def _placeholder1_btn3_cmd(self):
        pass

    def attach_bind(self):
        self._canvas.bind("<B1-Motion>", self._paint)
        self._canvas.bind("<Button-1>", self._start_paint)  # Bind left mouse button press to start a new line
        self._canvas.bind("<ButtonRelease-1>", self._reset_position)  # Bind left mouse button release to reset position
        self._canvas.bind("<Button-3>", self._reset_canvas)

    def _start_paint(self, event):
        # Set the starting point for the new line
        self._last_x, self._last_y = event.x, event.y

    def _paint(self, event):
        if self._last_x and self._last_y:
            self._canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=self.drawing_thickness,
                                     fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        # Update the last position to the current position
        self._last_x, self._last_y = event.x, event.y

    def _reset_position(self, event):
        # Reset the last position to prevent drawing a line from the last point when starting a new one
        self._last_x, self._last_y = None, None

    def _reset_canvas(self, event=None):  # default None for event to allow button use
        self._canvas.delete("all")
        self.last_x, self.last_y = None, None

    def _thickness_increase(self):
        if self.drawing_thickness != 10:
            self.drawing_thickness += 1

    def _thickness_decrease(self):
        if self.drawing_thickness != 1:
            self.drawing_thickness -= 1

    def push_drawing_knn(self):
        pass

    def push_drawing_logr(self):
        pass

    def push_drawing_nn(self):
        pass

    """def save_drawing(self):
        # Ensure the entire window layout is finalized
        self.root.update()  # This ensures the layout is complete

        # Calculate the absolute coordinates of the canvas
        x = self.root.winfo_x() + self.canvas_frame.winfo_x() + self.canvas.winfo_x() + 10
        y = self.root.winfo_y() + self.canvas_frame.winfo_y() + self.canvas.winfo_y() + 33
        x1 = x + self.canvas.winfo_width() - 4
        y1 = y + self.canvas.winfo_height() - 4

        # Grab the image, including only the canvas
        image = ImageGrab.grab().crop((x, y, x1, y1))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        data_instance = cv2.resize(cv_image, (126, 50), interpolation=cv2.INTER_AREA)

        # Save the image using a file dialog
        filepath = filedialog.asksaveasfilename(defaultextension='.png',
                                                filetypes=[("PNG files", '*.png'), ("JPEG files", '*.jpeg')])
        if filepath:
            cv2.imwrite(filepath, data_instance)"""


class redirect_print:
    def __init__(self, text_widget):
        self.output = text_widget

    def cprint(self, string, color='white'):
        self.output.insert(tk.END, string + '\n', color)
        self.output.see(tk.END)
