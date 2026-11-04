import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk

class MenuCaos:
    def __init__(self, root):
        self.root = root
        self.root.title("Simuladores de Sistemas Caóticos")
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.caos_path = os.path.join(base_dir, "caos_gui.py")
        self.lorenz_path = os.path.join(base_dir, "lorenz_gui.py")

        # Configuración visual
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        lbl_title = ttk.Label(main_frame, text="Sistemas Dinámicos y Caos", font=("Segoe UI", 16, "bold"))
        lbl_title.pack(pady=(0, 6))

        lbl_sub = ttk.Label(main_frame, text="Selecciona la simulación interactiva que deseas abrir:",
                             font=("Segoe UI", 9), foreground="#555")
        lbl_sub.pack(pady=(0, 15))

        # Tarjeta 1: Péndulo Doble
        f1 = ttk.LabelFrame(main_frame, text=" 🌀 Péndulo Doble ", padding=10)
        f1.pack(fill=tk.X, pady=6)
        ttk.Label(f1, text="Simulación del movimiento físico de péndulos acoplados en 2D.\n"
                           "Permite ver la trayectoria caótica y la divergencia con perturbación.",
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 6))
        ttk.Button(f1, text="Abrir Simulador de Péndulo Doble",
                   command=lambda: self.abrir_script(self.caos_path)).pack(fill=tk.X)

        # Tarjeta 2: Atractor de Lorenz 3D
        f2 = ttk.LabelFrame(main_frame, text=" 🦋 Atractor de Lorenz 3D ", padding=10)
        f2.pack(fill=tk.X, pady=6)
        ttk.Label(f2, text="Espacio de fase tridimensional del sistema convectivo de Lorenz.\n"
                           "Ajuste de σ, ρ, β, doble trayectoria (Efecto Mariposa) y rotación 3D.",
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 6))
        ttk.Button(f2, text="Abrir Simulador del Atractor de Lorenz 3D",
                   command=lambda: self.abrir_script(self.lorenz_path)).pack(fill=tk.X)

        # Botón para abrir ambos
        f_both = ttk.Frame(main_frame)
        f_both.pack(fill=tk.X, pady=(15, 0))
        ttk.Button(f_both, text="Abrir Ambos a la Vez", command=self.abrir_ambos).pack(fill=tk.X)

    def abrir_script(self, path):
        if os.path.exists(path):
            subprocess.Popen([sys.executable, path])

    def abrir_ambos(self):
        self.abrir_script(self.caos_path)
        self.abrir_script(self.lorenz_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = MenuCaos(root)
    root.mainloop()
