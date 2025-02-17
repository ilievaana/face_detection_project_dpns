import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageSequence
from modules.image_processing import process_image
from modules.video_processing import process_video


class AnimatedGIF:
    """A class for displaying a GIF animation in Tkinter."""
    def __init__(self, canvas, gif_path):
        self.canvas = canvas
        self.gif_path = gif_path
        self.frames = []
        self.load_gif()
        self.current_frame = 0

    def load_gif(self):
        gif = Image.open(self.gif_path)
        for frame in ImageSequence.Iterator(gif):
            frame = frame.resize((800, 600))
            self.frames.append(ImageTk.PhotoImage(frame))

    def animate(self):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frames[self.current_frame])
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.canvas.after(100, self.animate)


def main():
    def open_image():
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if filepath:
            process_image(filepath)
            messagebox.showinfo("Success", "Image processing completed!")
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def open_video():
        process_video()

    def on_enter(event):
        event.widget["bg"] = "#FFD700"

    def on_leave(event):
        event.widget["bg"] = event.widget.default_bg

    def toggle_instructions():
        if instructions_label.winfo_viewable():
            instructions_label.place_forget()
        else:
            instructions_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    root = tk.Tk()
    root.title("Face Detection Application")
    root.geometry("800x600")

    # Canvas for background animation
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack(fill=tk.BOTH, expand=True)

    # GIF animation
    gif_player = AnimatedGIF(canvas, "resources/background.gif")
    gif_player.animate()

    # Text
    tk.Label(
        root,
        text="Face Detection Application",
        font=("Helvetica", 24, "bold"),
        bg="#000000",
        fg="#FFFFFF"
    ).place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    # Buttons
    button_frame = tk.Frame(root, bg=root["bg"])
    button_frame.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    buttons = [
        ("Process Image", open_image, "#4CAF50"),
        ("Start Live Video", open_video, "#2196F3"),
    ]

    for i, (text, command, color) in enumerate(buttons):
        btn = tk.Button(
            button_frame,
            text=text,
            command=command,
            width=20,
            height=2,
            bg=color,
            fg="white",
            font=("Helvetica", 14),
            bd=0
        )
        btn.grid(row=i, column=0, padx=20, pady=10)
        btn.default_bg = color
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # Instructions
    instructions_label = tk.Label(
        root,
        text=(
            "1. Select an image for facial landmark detection.\n"
            "2. Or start live video for real-time analysis.\n"
            "3. Press 'q' to exit live video."
        ),
        font=("Verdana", 12),
        bg="#000000",
        fg="#FFFFFF",
        justify="center"
    )

    # Button for instructions
    instructions_btn = tk.Button(
        root,
        text="Read Instructions!",
        command=toggle_instructions,
        bg="#FFC300",
        fg="#000000",
        font=("Helvetica", 14, "bold"),
        width=20,
        height=2,
        bd=0
    )
    instructions_btn.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
    instructions_btn.default_bg = "#FFC300"
    instructions_btn.bind("<Enter>", on_enter)
    instructions_btn.bind("<Leave>", on_leave)

    root.mainloop()


if __name__ == "__main__":
    main()





