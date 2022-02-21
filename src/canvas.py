import tkinter as tk
from simulation import Simulation
from PIL import Image, ImageTk
import numpy as np



class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.canvas = None
        self.img = None
        self.prev_time = 0

        self.width = 500
        self.height = 500
        self.create_widgets()
        self.simulation = Simulation()
        self.cellWidth = self.width / self.simulation.width
        self.cellHeight = self.height / self.simulation.height

        self.fps = 20
        self.tps = 20
        self.after(int(1000 / self.tps), self.render_loop)
        self.after(int(1000 / self.tps), self.simulation_loop)

        # for i in range(10):
        #     for _ in range(0, 75):
        #         self.simulation.tick()
        #     print(f"Run {i}, fitness: {self.simulation.get_fitness()}")
        #     self.simulation.reset_grid()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, bg="#333", height=self.width, width=self.height)
        self.canvas.pack()

    def simulation_loop(self, i=0):
        self.simulation.tick()
        print(f"Tick {i}")
        self.after(int(1000 / self.tps), self.simulation_loop, i + 1)

    def render_loop(self, i=0):
        # current_time = time.time_ns()
        # elapsed = current_time - self.prev_time
        # self.prev_time = current_time
        # print(f"Fps: {round(1000000000 / elapsed, ndigits=1)}")

        self.canvas.delete("all")

        pil_image = Image.fromarray((self.simulation.grid[:, :, 0:3] * 255).astype(np.int8), "RGB")
        # pil_image = Image.fromarray(self.simulation.grid[:, :, 0] * 255)
        pil_image = pil_image.resize((self.width, self.height), resample=Image.NEAREST)
        self.img = ImageTk.PhotoImage(image=pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # draw_time = (time.time_ns() - current_time) // 1000000
        # call_after = int(1000 / self.fps) - draw_time
        call_after = int(1000 / self.fps)
        self.after(call_after, self.render_loop, i + 1)


if __name__ == '__main__':
    # create the application
    app = App()

    #
    # here are method calls to the window manager class
    #
    app.master.title("Simulation")
    app.master.maxsize(1900, 1000)
    app.master.geometry("900x900")
    app.master.configure(bg='#111')

    # start the program
    app.mainloop()
