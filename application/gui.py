import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from tkinter import *
from tkinter import filedialog, messagebox
from keras.models import load_model


class SegmentationGUI:
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = "black"

    def __init__(self):
        self.root = Tk()
        self.root.title("Interactive Segmentation Tool")
        self.root.geometry("1300x800")

        # Load pre-trained model
        self.model = load_model(
            r"C:\Users\krish\OneDrive\Desktop\personal_growth\Skin-Lesion\application\outputs\unet_segmentation_model.h5",
            compile=False,
        )

        # Title
        Label(
            self.root,
            text="Self-Adapting Interactive Segmentation Tool",
            width=100,
            height=2,
            fg="black",
            font=("times", 20, "bold"),
        ).pack()

        # Buttons
        Button(
            self.root,
            text="Select Image",
            command=self.choose_image,
            bg="grey",
            fg="lightyellow",
            width=12,
            height=2,
            font=("times", 15, "bold"),
        ).place(x=30, y=80)
        Button(
            self.root,
            text="Segment",
            command=self.segment_image,
            bg="grey",
            fg="lightyellow",
            width=12,
            height=2,
            font=("times", 15, "bold"),
        ).place(x=30, y=150)
        Button(
            self.root,
            text="Save Result",
            command=self.save_result,
            bg="grey",
            fg="lightyellow",
            width=12,
            height=2,
            font=("times", 15, "bold"),
        ).place(x=30, y=220)
        Button(
            self.root,
            text="Pen",
            command=self.use_pen,
            bg="grey",
            fg="lightyellow",
            width=12,
            height=2,
            font=("times", 15, "bold"),
        ).place(x=30, y=290)
        Button(
            self.root,
            text="Eraser",
            command=self.use_eraser,
            bg="grey",
            fg="lightyellow",
            width=12,
            height=2,
            font=("times", 15, "bold"),
        ).place(x=30, y=360)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.set(5)
        self.choose_size_button.place(x=30, y=430)

        self.canvas = Canvas(self.root, width=650, height=650, bg="white")
        self.canvas.place(x=400, y=100)

        self.old_x = None
        self.old_y = None
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.image_path = None
        self.overlay = None
        self.image_on_canvas = None

        self.root.mainloop()

    def choose_image(self):
        file = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if file:
            self.image_path = file
            self.display_image(Image.open(file))

    def display_image(self, img):
        self.img = img.resize((450, 450)).convert("RGB")
        self.tk_image = ImageTk.PhotoImage(self.img)
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=NW, image=self.tk_image
        )
        self.overlay = Image.new("RGBA", self.img.size)
        self.draw = ImageDraw.Draw(self.overlay)

    def segment_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0  # normalize if model expects it

        pred = self.model.predict(image[np.newaxis, ...])
        pred_mask = (pred > 0.5).astype(np.uint8)[0, :, :, 0] * 255
        pred_img = Image.fromarray(pred_mask).resize((450, 450)).convert("RGBA")

        # Make white regions transparent (optional: adjust thresholding)
        datas = pred_img.getdata()
        transparent_data = [
            (r, g, b, 128) if r > 0 else (r, g, b, 0) for r, g, b, a in datas
        ]
        pred_img.putdata(transparent_data)

        self.overlay = pred_img
        final_img = Image.alpha_composite(self.img.convert("RGBA"), pred_img)
        self.tk_image = ImageTk.PhotoImage(final_img)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

    def save_result(self):
        if self.overlay:
            result = Image.alpha_composite(self.img.convert("RGBA"), self.overlay)
            output_path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG files", "*.png")]
            )
            if output_path:
                result.save(output_path)
                messagebox.showinfo(
                    "Saved", f"Segmented output saved to:\n{output_path}"
                )

    def use_pen(self):
        self.activate_button("pen")

    def use_eraser(self):
        self.activate_button("eraser")

    def activate_button(self, tool):
        self.eraser_on = tool == "eraser"
        self.active_button = tool

    def paint(self, event):
        paint_color = "white" if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.choose_size_button.get(),
                fill=paint_color,
                capstyle=ROUND,
                smooth=TRUE,
            )
            self.draw.line(
                (self.old_x, self.old_y, event.x, event.y),
                fill=paint_color,
                width=self.choose_size_button.get(),
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == "__main__":
    SegmentationGUI()
