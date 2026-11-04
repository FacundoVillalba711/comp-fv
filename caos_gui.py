import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# --- Ecuaciones del Péndulo Doble ---
def pendulo_doble_ode(t, y, m1, m2, l1, l2, g):
    th1, w1, th2, w2 = y
    delta = th1 - th2

    M = np.array([
        [(m1 + m2) * l1**2,           m2 * l1 * l2 * np.cos(delta)],
        [m2 * l1 * l2 * np.cos(delta), m2 * l2**2]
    ])

    b = np.array([
        -m2 * l1 * l2 * (w2**2) * np.sin(delta) - (m1 + m2) * g * l1 * np.sin(th1),
         m2 * l1 * l2 * (w1**2) * np.sin(delta) - m2 * g * l2 * np.sin(th2)
    ])

    alpha1, alpha2 = np.linalg.solve(M, b)
    return [w1, alpha1, w2, alpha2]

class AppPenduloDoble:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Interactivo: Péndulos Dobles y Caos")
        self.root.geometry("1100x750")
        self.root.minsize(950, 650)

        # Estado de animación
        self.anim_running = False
        self.anim_job = None
        self.frame_idx = 0
        self.total_frames = 0
        self.t_eval = None
        self.data_A = None
        self.data_B = None

        self.setup_ui()
        self.setup_plot()
        # Simulación inicial por defecto
        self.ejecutar_simulacion()

    def setup_ui(self):
        # Panel principal dividido: Izquierda (controles) y Derecha (gráfico)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- PANEL LATERAL DE CONTROLES ---
        self.control_frame = ttk.Frame(self.main_paned, padding=10)
        self.main_paned.add(self.control_frame, weight=0)

        # Título
        lbl_title = ttk.Label(self.control_frame, text="Parámetros del Sistema", font=("Segoe UI", 12, "bold"))
        lbl_title.pack(anchor="w", pady=(0, 10))

        # Helper para crear entradas con etiqueta
        self.entries = {}
        def add_entry(parent, text, default_val, key):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=3)
            lbl = ttk.Label(f, text=text, width=22, anchor="w")
            lbl.pack(side=tk.LEFT)
            ent = ttk.Entry(f, width=12)
            ent.insert(0, str(default_val))
            ent.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.entries[key] = ent

        # 1. Parámetros Físicos
        grp_fisica = ttk.LabelFrame(self.control_frame, text="Propiedades Físicas", padding=8)
        grp_fisica.pack(fill=tk.X, pady=(0, 10))
        add_entry(grp_fisica, "Longitud 1 (m):", 1.0, "l1")
        add_entry(grp_fisica, "Longitud 2 (m):", 1.0, "l2")
        add_entry(grp_fisica, "Masa 1 (kg):", 1.0, "m1")
        add_entry(grp_fisica, "Masa 2 (kg):", 1.0, "m2")
        add_entry(grp_fisica, "Gravedad (m/s²):", 9.81, "g")

        # 2. Condiciones Iniciales (Péndulo A)
        grp_ci = ttk.LabelFrame(self.control_frame, text="Condiciones Iniciales (Péndulo A)", padding=8)
        grp_ci.pack(fill=tk.X, pady=(0, 10))
        add_entry(grp_ci, "Ángulo θ1 (°):", 90.0, "th1_deg")
        add_entry(grp_ci, "Ángulo θ2 (°):", 90.0, "th2_deg")
        add_entry(grp_ci, "Vel. angular ω1 (°/s):", 0.0, "w1_deg")
        add_entry(grp_ci, "Vel. angular ω2 (°/s):", 0.0, "w2_deg")

        # 3. Diferencia para Péndulo B
        grp_caos = ttk.LabelFrame(self.control_frame, text="Perturbación (Péndulo B)", padding=8)
        grp_caos.pack(fill=tk.X, pady=(0, 10))
        add_entry(grp_caos, "Diferencia Δθ1 (°):", 0.005, "delta_deg")
        lbl_hint = ttk.Label(grp_caos, text="(Ej: 0.005° genera divergencia caótica rápida)", font=("Segoe UI", 8), foreground="#666")
        lbl_hint.pack(anchor="w", pady=(2, 0))

        # 4. Opciones de Simulación
        grp_sim = ttk.LabelFrame(self.control_frame, text="Opciones de Simulación", padding=8)
        grp_sim.pack(fill=tk.X, pady=(0, 10))
        add_entry(grp_sim, "Tiempo total (s):", 20.0, "t_max")
        add_entry(grp_sim, "FPS animación:", 30, "fps")

        self.var_estela_permanente = tk.BooleanVar(value=True)
        chk_estela = ttk.Checkbutton(grp_sim, text="Estela permanente", variable=self.var_estela_permanente)
        chk_estela.pack(anchor="w", pady=4)

        # 5. Botones de Acción
        grp_botones = ttk.Frame(self.control_frame)
        grp_botones.pack(fill=tk.X, pady=10)

        self.btn_simular = ttk.Button(grp_botones, text="▶ Simular y Reproducir", command=self.ejecutar_simulacion)
        self.btn_simular.pack(fill=tk.X, pady=3)

        self.btn_pausa = ttk.Button(grp_botones, text="⏸ Pausar", command=self.toggle_pausa)
        self.btn_pausa.pack(fill=tk.X, pady=3)

        self.btn_reiniciar = ttk.Button(grp_botones, text="↺ Reiniciar Vista", command=self.reiniciar_animacion)
        self.btn_reiniciar.pack(fill=tk.X, pady=3)

        self.btn_gif = ttk.Button(grp_botones, text="💾 Exportar animación a GIF", command=self.exportar_gif)
        self.btn_gif.pack(fill=tk.X, pady=(8, 3))

        # Barra de estado
        self.lbl_status = ttk.Label(self.control_frame, text="Listo.", relief=tk.SUNKEN, anchor="w", padding=4)
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_plot(self):
        # Panel derecho para el canvas de matplotlib
        self.plot_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.plot_frame, weight=1)

        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        # Artistas gráficos: Péndulo A (azul), Péndulo B (naranja)
        self.linea_brazos_A, = self.ax.plot([], [], 'o-', lw=2.2, color='#1f77b4', markersize=7, label="Péndulo A")
        self.linea_estela_A, = self.ax.plot([], [], '-', lw=0.9, color='#1f77b4', alpha=0.6)

        self.linea_brazos_B, = self.ax.plot([], [], 'o-', lw=2.2, color='#ff7f0e', markersize=7, label="Péndulo B")
        self.linea_estela_B, = self.ax.plot([], [], '-', lw=0.9, color='#ff7f0e', alpha=0.6)

        self.ax.plot(0, 0, 'ko', markersize=6)  # Pivote
        self.txt_tiempo = self.ax.text(0.03, 0.95, "", transform=self.ax.transAxes, fontfamily="monospace", fontsize=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def leer_parametros(self):
        try:
            p = {
                "l1": float(self.entries["l1"].get()),
                "l2": float(self.entries["l2"].get()),
                "m1": float(self.entries["m1"].get()),
                "m2": float(self.entries["m2"].get()),
                "g": float(self.entries["g"].get()),
                "th1_deg": float(self.entries["th1_deg"].get()),
                "th2_deg": float(self.entries["th2_deg"].get()),
                "w1_deg": float(self.entries["w1_deg"].get()),
                "w2_deg": float(self.entries["w2_deg"].get()),
                "delta_deg": float(self.entries["delta_deg"].get()),
                "t_max": float(self.entries["t_max"].get()),
                "fps": max(5, int(float(self.entries["fps"].get())))
            }
            if p["l1"] <= 0 or p["l2"] <= 0 or p["m1"] <= 0 or p["m2"] <= 0 or p["t_max"] <= 0:
                raise ValueError("Las longitudes, masas y tiempo deben ser números positivos.")
            return p
        except ValueError as err:
            messagebox.showerror("Error de entrada", f"Valor inválido:\n{err}")
            return None

    def ejecutar_simulacion(self):
        # Detener animación previa
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.anim_running = False

        p = self.leer_parametros()
        if not p:
            return

        self.lbl_status.config(text="Integrando ecuaciones numéricas...")
        self.root.update_idletasks()

        # Condiciones iniciales en radianes
        th1_A = np.radians(p["th1_deg"])
        th2_A = np.radians(p["th2_deg"])
        w1_A = np.radians(p["w1_deg"])
        w2_A = np.radians(p["w2_deg"])
        y0_A = [th1_A, w1_A, th2_A, w2_A]

        delta_rad = np.radians(p["delta_deg"])
        y0_B = [th1_A + delta_rad, w1_A, th2_A, w2_A]

        # Vector temporal
        fps = p["fps"]
        self.total_frames = int(p["t_max"] * fps)
        self.t_eval = np.linspace(0, p["t_max"], self.total_frames)

        # Resolver numéricamente con SciPy
        sol_A = solve_ivp(
            pendulo_doble_ode,
            t_span=(0, p["t_max"]),
            y0=y0_A,
            t_eval=self.t_eval,
            method="RK45",
            args=(p["m1"], p["m2"], p["l1"], p["l2"], p["g"]),
            rtol=1e-9,
            atol=1e-9
        )

        sol_B = solve_ivp(
            pendulo_doble_ode,
            t_span=(0, p["t_max"]),
            y0=y0_B,
            t_eval=self.t_eval,
            method="RK45",
            args=(p["m1"], p["m2"], p["l1"], p["l2"], p["g"]),
            rtol=1e-9,
            atol=1e-9
        )

        # Calcular coordenadas cartesianas
        l1, l2 = p["l1"], p["l2"]

        # Péndulo A
        th1_vals_A, th2_vals_A = sol_A.y[0], sol_A.y[2]
        x1_A = l1 * np.sin(th1_vals_A)
        y1_A = -l1 * np.cos(th1_vals_A)
        x2_A = x1_A + l2 * np.sin(th2_vals_A)
        y2_A = y1_A - l2 * np.cos(th2_vals_A)

        # Péndulo B
        th1_vals_B, th2_vals_B = sol_B.y[0], sol_B.y[2]
        x1_B = l1 * np.sin(th1_vals_B)
        y1_B = -l1 * np.cos(th1_vals_B)
        x2_B = x1_B + l2 * np.sin(th2_vals_B)
        y2_B = y1_B - l2 * np.cos(th2_vals_B)

        self.data_A = (x1_A, y1_A, x2_A, y2_A)
        self.data_B = (x1_B, y1_B, x2_B, y2_B)

        # Ajustar límites de los ejes
        L = (l1 + l2) * 1.15
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_title(f"Péndulo Doble: A vs B (Δθ₁ = {p['delta_deg']}°)")
        self.linea_brazos_A.set_label(f"Péndulo A (θ₁={p['th1_deg']}°)")
        self.linea_brazos_B.set_label(f"Péndulo B (+{p['delta_deg']}°)")
        self.ax.legend(loc="upper right")

        self.frame_idx = 0
        self.anim_running = True
        self.btn_pausa.config(text="⏸ Pausar")
        self.lbl_status.config(text=f"Simulación lista. Reproduciendo ({self.total_frames} frames)...")

        # Iniciar ciclo de reproducción
        self.frame_delay_ms = int(1000 / fps)
        self.animar_frame()

    def animar_frame(self):
        if not self.anim_running or self.data_A is None:
            return

        i = self.frame_idx
        x1_A, y1_A, x2_A, y2_A = self.data_A
        x1_B, y1_B, x2_B, y2_B = self.data_B

        # Actualizar brazos
        self.linea_brazos_A.set_data([0, x1_A[i], x2_A[i]], [0, y1_A[i], y2_A[i]])
        self.linea_brazos_B.set_data([0, x1_B[i], x2_B[i]], [0, y1_B[i], y2_B[i]])

        # Actualizar estelas
        if self.var_estela_permanente.get():
            self.linea_estela_A.set_data(x2_A[:i + 1], y2_A[:i + 1])
            self.linea_estela_B.set_data(x2_B[:i + 1], y2_B[:i + 1])
        else:
            trail_len = max(1, int(len(self.t_eval) * 0.08))
            start_idx = max(0, i - trail_len)
            self.linea_estela_A.set_data(x2_A[start_idx:i + 1], y2_A[start_idx:i + 1])
            self.linea_estela_B.set_data(x2_B[start_idx:i + 1], y2_B[start_idx:i + 1])

        # Tiempo actual
        t_actual = self.t_eval[i]
        self.txt_tiempo.set_text(f"t = {t_actual:05.2f} s / frame {i+1}/{self.total_frames}")

        self.canvas.draw_idle()

        # Avanzar al siguiente fotograma
        self.frame_idx += 1
        if self.frame_idx >= self.total_frames:
            self.anim_running = False
            self.btn_pausa.config(text="▶ Reproducir")
            self.lbl_status.config(text="Animación completada.")
        else:
            self.anim_job = self.root.after(self.frame_delay_ms, self.animar_frame)

    def toggle_pausa(self):
        if self.data_A is None:
            return
        if self.anim_running:
            self.anim_running = False
            if self.anim_job:
                self.root.after_cancel(self.anim_job)
                self.anim_job = None
            self.btn_pausa.config(text="▶ Reanudar")
            self.lbl_status.config(text="Animación pausada.")
        else:
            if self.frame_idx >= self.total_frames:
                self.frame_idx = 0
            self.anim_running = True
            self.btn_pausa.config(text="⏸ Pausar")
            self.lbl_status.config(text="Reproduciendo...")
            self.animar_frame()

    def reiniciar_animacion(self):
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.frame_idx = 0
        if self.data_A is not None:
            self.anim_running = True
            self.btn_pausa.config(text="⏸ Pausar")
            self.animar_frame()

    def exportar_gif(self):
        if self.data_A is None:
            messagebox.showinfo("Aviso", "Primero ejecuta una simulación para poder exportar.")
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF Image", "*.gif")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
            initialfile="pendulo_doble_interactivo.gif",
            title="Guardar animación como GIF"
        )
        if not ruta:
            return

        # Pausar mientras se guarda
        if self.anim_running:
            self.toggle_pausa()

        self.lbl_status.config(text="Generando archivo GIF... Por favor espere.")
        self.root.update_idletasks()

        try:
            x1_A, y1_A, x2_A, y2_A = self.data_A
            x1_B, y1_B, x2_B, y2_B = self.data_B
            p = self.leer_parametros()
            fps = p["fps"]

            fig_gif = Figure(figsize=(7, 7), dpi=80)
            ax_gif = fig_gif.add_subplot(111)
            L = (p["l1"] + p["l2"]) * 1.15
            ax_gif.set_xlim(-L, L)
            ax_gif.set_ylim(-L, L)
            ax_gif.set_aspect("equal")
            ax_gif.grid(True, alpha=0.3)
            ax_gif.set_title(f"Caos: Péndulo Doble (Δθ = {p['delta_deg']}°)")

            b_A, = ax_gif.plot([], [], 'o-', lw=2.2, color='#1f77b4', markersize=6, label="Péndulo A")
            e_A, = ax_gif.plot([], [], '-', lw=0.8, color='#1f77b4', alpha=0.5)
            b_B, = ax_gif.plot([], [], 'o-', lw=2.2, color='#ff7f0e', markersize=6, label=f"Péndulo B (+{p['delta_deg']}°)")
            e_B, = ax_gif.plot([], [], '-', lw=0.8, color='#ff7f0e', alpha=0.5)
            ax_gif.plot(0, 0, 'ko', markersize=5)
            ax_gif.legend(loc="upper right")

            def update_gif(f):
                b_A.set_data([0, x1_A[f], x2_A[f]], [0, y1_A[f], y2_A[f]])
                b_B.set_data([0, x1_B[f], x2_B[f]], [0, y1_B[f], y2_B[f]])
                if self.var_estela_permanente.get():
                    e_A.set_data(x2_A[:f + 1], y2_A[:f + 1])
                    e_B.set_data(x2_B[:f + 1], y2_B[:f + 1])
                else:
                    t_len = max(1, int(len(self.t_eval) * 0.08))
                    st = max(0, f - t_len)
                    e_A.set_data(x2_A[st:f + 1], y2_A[st:f + 1])
                    e_B.set_data(x2_B[st:f + 1], y2_B[st:f + 1])
                return b_A, e_A, b_B, e_B

            # Submuestreo si hay muchos frames para agilizar la exportación
            step = max(1, int(self.total_frames / (p["t_max"] * 25)))
            frames_to_save = range(0, self.total_frames, step)

            anim = animation.FuncAnimation(fig_gif, update_gif, frames=frames_to_save, blit=True)
            anim.save(ruta, writer="pillow", fps=min(fps, 25))

            self.lbl_status.config(text=f"¡GIF guardado exitosamente en {os.path.basename(ruta)}!")
            messagebox.showinfo("Exportación Exitosa", f"El GIF se ha guardado en:\n{ruta}")
        except Exception as ex:
            self.lbl_status.config(text="Error al exportar GIF.")
            messagebox.showerror("Error al guardar", f"No se pudo guardar el GIF:\n{ex}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppPenduloDoble(root)
    root.mainloop()
