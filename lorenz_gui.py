import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# ECUACIONES DEL SISTEMA DINÁMICO DE LORENZ
# ==============================================================================
# dx/dt = sigma * (y - x)
# dy/dt = x * (rho - z) - y
# dz/dt = x * y - beta * z
# ==============================================================================

def lorenz_ode(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


class ScrollableFrame(ttk.Frame):
    """Contenedor scrollable vertical para controles en pantallas de cualquier tamaño."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_content = ttk.Frame(self.canvas)

        self.scrollable_content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class AppAtractorLorenz:
    PRESETS = {
        "Caos clásico (Mariposa) [σ=10, ρ=28, β=8/3]": {
            "sigma": 10.0, "rho": 28.0, "beta": 2.6667,
            "x0": 1.0, "y0": 1.0, "z0": 1.0, "dx": 0.001, "t_max": 40.0
        },
        "Puntos fijos estables [σ=10, ρ=14, β=8/3]": {
            "sigma": 10.0, "rho": 14.0, "beta": 2.6667,
            "x0": 1.0, "y0": 1.0, "z0": 1.0, "dx": 0.001, "t_max": 30.0
        },
        "Ciclo límite periódico [σ=10, ρ=350, β=8/3]": {
            "sigma": 10.0, "rho": 350.0, "beta": 2.6667,
            "x0": 5.0, "y0": 5.0, "z0": 20.0, "dx": 0.001, "t_max": 20.0
        },
        "Caos intermitente [σ=10, ρ=99.96, β=8/3]": {
            "sigma": 10.0, "rho": 99.96, "beta": 2.6667,
            "x0": 1.0, "y0": 1.0, "z0": 1.0, "dx": 0.001, "t_max": 30.0
        },
        "Estructura laminar [σ=10, ρ=166.3, β=8/3]": {
            "sigma": 10.0, "rho": 166.3, "beta": 2.6667,
            "x0": 1.0, "y0": 1.0, "z0": 1.0, "dx": 0.001, "t_max": 25.0
        },
        "Transitorio caótico largo [σ=10, ρ=23.5, β=8/3]": {
            "sigma": 10.0, "rho": 23.5, "beta": 2.6667,
            "x0": 1.0, "y0": 1.0, "z0": 1.0, "dx": 0.001, "t_max": 35.0
        }
    }

    THEMES = {
        "oscuro": {
            "fig_bg": "#0d1117",
            "ax_bg": "#0d1117",
            "pane_color": (0.07, 0.09, 0.13, 0.7),
            "grid_color": "#21262d",
            "text_color": "#c9d1d9",
            "line_a": "#00f0ff",      # Cyan neón
            "head_a": "#ffffff",
            "line_b": "#ff4d4d",      # Rojo/Coral neón
            "head_b": "#ffdf00",
            "fp_origin": "#8b949e",
            "fp_foci": "#e3b341"
        },
        "claro": {
            "fig_bg": "#ffffff",
            "ax_bg": "#ffffff",
            "pane_color": (0.95, 0.95, 0.97, 0.7),
            "grid_color": "#e1e4e8",
            "text_color": "#24292e",
            "line_a": "#0969da",      # Azul intenso
            "head_a": "#043263",
            "line_b": "#cf222e",      # Rojo intenso
            "head_b": "#82071e",
            "fp_origin": "#57606a",
            "fp_foci": "#bf8700"
        }
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Interactivo: Atractor de Lorenz (Espacio de Fase 3D)")
        self.root.geometry("1200x820")
        self.root.minsize(1000, 700)

        # Variables de estado de simulación
        self.anim_running = False
        self.anim_job = None
        self.current_idx = 0
        self.total_points = 0
        self.t_eval = None
        self.data_A = None  # (x, y, z)
        self.data_B = None  # (x, y, z)
        self.p_actual = None
        self.user_dragging_slider = False

        # Configuración visual actual
        self.current_theme_name = "oscuro"

        # Construcción de la interfaz
        self.setup_ui()
        self.setup_plot()

        # Cargar simulación inicial por defecto
        self.ejecutar_simulacion()

    def setup_ui(self):
        # Panel principal dividido: Izquierda (controles) y Derecha (gráfico 3D)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- PANEL LATERAL DE CONTROLES CON SCROLL ---
        self.scroll_container = ScrollableFrame(self.main_paned, width=380)
        self.main_paned.add(self.scroll_container, weight=0)
        self.control_frame = self.scroll_container.scrollable_content

        # Título del panel
        lbl_title = ttk.Label(self.control_frame, text="Atractor de Lorenz 3D", font=("Segoe UI", 13, "bold"))
        lbl_title.pack(anchor="w", padx=4, pady=(2, 4))
        lbl_desc = ttk.Label(self.control_frame, text="Exploración del espacio de fase y caos determinista",
                             font=("Segoe UI", 8), foreground="#666")
        lbl_desc.pack(anchor="w", padx=4, pady=(0, 8))

        # Helper para crear entradas con etiqueta
        self.entries = {}
        def add_entry(parent, text, default_val, key, tooltip=""):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(f, text=text, width=22, anchor="w")
            lbl.pack(side=tk.LEFT)
            ent = ttk.Entry(f, width=11)
            ent.insert(0, str(default_val))
            ent.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.entries[key] = ent
            if tooltip:
                tip = ttk.Label(parent, text=tooltip, font=("Segoe UI", 7), foreground="#777")
                tip.pack(anchor="w", pady=(0, 2))

        # 0. Menú de Configuraciones Predefinidas (Presets)
        grp_presets = ttk.LabelFrame(self.control_frame, text="Configuraciones Teóricas (Presets)", padding=6)
        grp_presets.pack(fill=tk.X, pady=(0, 6), padx=4)

        self.cb_preset = ttk.Combobox(grp_presets, values=list(self.PRESETS.keys()), state="readonly")
        self.cb_preset.current(0)
        self.cb_preset.pack(fill=tk.X, pady=2)
        self.cb_preset.bind("<<ComboboxSelected>>", self.cargar_preset)

        # 1. Parámetros del Atractor de Lorenz
        grp_params = ttk.LabelFrame(self.control_frame, text="Parámetros Físicos del Sistema", padding=6)
        grp_params.pack(fill=tk.X, pady=(0, 6), padx=4)
        add_entry(grp_params, "σ (Sigma, Prandtl):", 10.0, "sigma", "Razón de viscosidad vs conducción térmica")
        add_entry(grp_params, "ρ (Rho, Rayleigh):", 28.0, "rho", "Gradiente de temperatura (>24.74 genera caos)")
        add_entry(grp_params, "β (Beta, Geometría):", "8/3", "beta", "Relación de aspecto de las celdas convectivas")

        # 2. Condiciones Iniciales (Trayectoria A)
        grp_ci = ttk.LabelFrame(self.control_frame, text="Condición Inicial: Trayectoria A", padding=6)
        grp_ci.pack(fill=tk.X, pady=(0, 6), padx=4)
        add_entry(grp_ci, "x₀ (Convección):", 1.0, "x0")
        add_entry(grp_ci, "y₀ (Gradiente temp.):", 1.0, "y0")
        add_entry(grp_ci, "z₀ (Distorsión vert.):", 1.0, "z0")

        # 3. Segunda Trayectoria y Efecto Mariposa (Trayectoria B)
        grp_mariposa = ttk.LabelFrame(self.control_frame, text="Doble Trayectoria (Efecto Mariposa)", padding=6)
        grp_mariposa.pack(fill=tk.X, pady=(0, 6), padx=4)

        self.var_comparar_b = tk.BooleanVar(value=True)
        chk_b = ttk.Checkbutton(grp_mariposa, text="Activar Trayectoria B (Perturbada)",
                                variable=self.var_comparar_b, command=self.toggle_trayectoria_b_ui)
        chk_b.pack(anchor="w", pady=(0, 4))

        self.frame_b_entries = ttk.Frame(grp_mariposa)
        self.frame_b_entries.pack(fill=tk.X)
        add_entry(self.frame_b_entries, "Perturbación Δx₀:", 0.001, "delta_x", "Diferencia minúscula inicial para observar divergencia")
        add_entry(self.frame_b_entries, "Perturbación Δy₀:", 0.0, "delta_y")
        add_entry(self.frame_b_entries, "Perturbación Δz₀:", 0.0, "delta_z")

        # 4. Opciones de Simulación y Visualización
        grp_sim = ttk.LabelFrame(self.control_frame, text="Simulación y Cámara 3D", padding=6)
        grp_sim.pack(fill=tk.X, pady=(0, 6), padx=4)
        add_entry(grp_sim, "Tiempo total T (s):", 35.0, "t_max")
        add_entry(grp_sim, "Paso temporal dt (s):", 0.01, "dt", "Menor dt da mayor precisión")

        # Velocidad de animación
        f_spd = ttk.Frame(grp_sim)
        f_spd.pack(fill=tk.X, pady=2)
        ttk.Label(f_spd, text="Velocidad anim.:", width=22, anchor="w").pack(side=tk.LEFT)
        self.cb_velocidad = ttk.Combobox(f_spd, values=["1x", "2x", "4x", "8x", "16x"], width=9, state="readonly")
        self.cb_velocidad.set("4x")
        self.cb_velocidad.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.var_auto_rotar = tk.BooleanVar(value=True)
        chk_rot = ttk.Checkbutton(grp_sim, text="Auto-rotación 3D continua", variable=self.var_auto_rotar)
        chk_rot.pack(anchor="w", pady=(3, 2))

        self.var_mostrar_focos = tk.BooleanVar(value=True)
        chk_focos = ttk.Checkbutton(grp_sim, text="Mostrar puntos de equilibrio (focos)",
                                    variable=self.var_mostrar_focos, command=self.actualizar_focos)
        chk_focos.pack(anchor="w", pady=(0, 2))

        # Selector de Tema
        f_thm = ttk.Frame(grp_sim)
        f_thm.pack(fill=tk.X, pady=3)
        ttk.Label(f_thm, text="Tema visual 3D:", width=22, anchor="w").pack(side=tk.LEFT)
        self.cb_tema = ttk.Combobox(f_thm, values=["Modo Oscuro (Neón)", "Modo Claro (Científico)"], width=9, state="readonly")
        self.cb_tema.current(0)
        self.cb_tema.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.cb_tema.bind("<<ComboboxSelected>>", self.cambiar_tema)

        # Vistas de Cámara Rápidas
        grp_vistas = ttk.LabelFrame(self.control_frame, text="Vistas Rápidas de Cámara", padding=6)
        grp_vistas.pack(fill=tk.X, pady=(0, 6), padx=4)
        f_vbtns = ttk.Frame(grp_vistas)
        f_vbtns.pack(fill=tk.X)
        ttk.Button(f_vbtns, text="3D Iso", width=7, command=lambda: self.fijar_vista(20, 45)).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        ttk.Button(f_vbtns, text="Frontal (X-Z)", width=9, command=lambda: self.fijar_vista(0, -90)).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        ttk.Button(f_vbtns, text="Superior (X-Y)", width=9, command=lambda: self.fijar_vista(90, -90)).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        ttk.Button(f_vbtns, text="Lateral (Y-Z)", width=9, command=lambda: self.fijar_vista(0, 0)).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)

        # 5. Barra de Tiempo Deslizante (Timeline Scrubber)
        grp_time = ttk.LabelFrame(self.control_frame, text="Línea de Tiempo", padding=6)
        grp_time.pack(fill=tk.X, pady=(0, 6), padx=4)
        self.slider_tiempo = ttk.Scale(grp_time, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.slider_tiempo.pack(fill=tk.X, pady=2)
        self.slider_tiempo.bind("<ButtonPress-1>", lambda e: self.on_slider_press())
        self.slider_tiempo.bind("<ButtonRelease-1>", lambda e: self.on_slider_release())

        # 6. Botones de Acción
        grp_botones = ttk.Frame(self.control_frame)
        grp_botones.pack(fill=tk.X, pady=6, padx=4)

        self.btn_simular = ttk.Button(grp_botones, text="▶ Recalcular y Reproducir", command=self.ejecutar_simulacion)
        self.btn_simular.pack(fill=tk.X, pady=2)

        f_anim_btns = ttk.Frame(grp_botones)
        f_anim_btns.pack(fill=tk.X, pady=2)
        self.btn_pausa = ttk.Button(f_anim_btns, text="⏸ Pausar", command=self.toggle_pausa)
        self.btn_pausa.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.btn_reiniciar = ttk.Button(f_anim_btns, text="↺ Reiniciar", command=self.reiniciar_animacion)
        self.btn_reiniciar.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))

        self.btn_dibujar_todo = ttk.Button(grp_botones, text="⚡ Ver Atractor Completo", command=self.dibujar_completo)
        self.btn_dibujar_todo.pack(fill=tk.X, pady=2)

        f_export = ttk.Frame(grp_botones)
        f_export.pack(fill=tk.X, pady=(4, 2))
        self.btn_img = ttk.Button(f_export, text="📷 Guardar Foto", command=self.exportar_imagen)
        self.btn_img.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.btn_gif = ttk.Button(f_export, text="🎥 Exportar GIF", command=self.exportar_gif)
        self.btn_gif.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))

        # Barra de estado inferior izquierda
        self.lbl_telemetria = ttk.Label(self.control_frame, text="Iniciando...", font=("Consolas", 8),
                                        relief=tk.SUNKEN, padding=4, justify=tk.LEFT)
        self.lbl_telemetria.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(6, 2))

    def setup_plot(self):
        # Panel derecho para el canvas 3D interactivo
        self.plot_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.plot_frame, weight=1)

        theme = self.THEMES[self.current_theme_name]

        self.fig = Figure(figsize=(8, 7), dpi=100, facecolor=theme["fig_bg"])
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(theme["ax_bg"])

        # Canvas de Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)

        # Barra de herramientas estándar de matplotlib (Zoom, Pan, Save, etc.)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configurar estilos de los ejes 3D
        self.aplicar_estilo_ejes()

        # Elementos gráficos de la trayectoria:
        # Trayectoria A
        self.line_A, = self.ax.plot([], [], [], lw=1.2, color=theme["line_a"], label="Trayectoria A", alpha=0.9)
        self.head_A, = self.ax.plot([], [], [], 'o', color=theme["head_a"], markersize=6)

        # Trayectoria B (perturbada)
        self.line_B, = self.ax.plot([], [], [], lw=1.2, color=theme["line_b"], label="Trayectoria B", alpha=0.9)
        self.head_B, = self.ax.plot([], [], [], 'o', color=theme["head_b"], markersize=6)

        # Puntos de equilibrio (Focos)
        self.sc_origin, = self.ax.plot([], [], [], 'd', color=theme["fp_origin"], markersize=7, label="Origen (0,0,0)")
        self.sc_foci, = self.ax.plot([], [], [], '*', color=theme["fp_foci"], markersize=9, label="Focos C±")

        # Texto de información en pantalla (HUD)
        self.txt_hud = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes,
                                      fontfamily="monospace", fontsize=9, color=theme["text_color"])

        self.ax.view_init(elev=20, azim=45)

    def aplicar_estilo_ejes(self):
        theme = self.THEMES[self.current_theme_name]
        self.fig.set_facecolor(theme["fig_bg"])
        self.ax.set_facecolor(theme["ax_bg"])

        # Paneles y rejillas 3D
        try:
            self.ax.xaxis.set_pane_color(theme["pane_color"])
            self.ax.yaxis.set_pane_color(theme["pane_color"])
            self.ax.zaxis.set_pane_color(theme["pane_color"])
        except Exception:
            pass

        self.ax.grid(True, color=theme["grid_color"], alpha=0.5, linestyle=":")
        self.ax.xaxis._axinfo["grid"]["color"] = theme["grid_color"]
        self.ax.yaxis._axinfo["grid"]["color"] = theme["grid_color"]
        self.ax.zaxis._axinfo["grid"]["color"] = theme["grid_color"]

        self.ax.set_xlabel("X (Convección)", color=theme["text_color"], fontsize=9, labelpad=8)
        self.ax.set_ylabel("Y (Gradiente T)", color=theme["text_color"], fontsize=9, labelpad=8)
        self.ax.set_zlabel("Z (Distorsión T)", color=theme["text_color"], fontsize=9, labelpad=8)

        self.ax.tick_params(colors=theme["text_color"], labelsize=8)

    def toggle_trayectoria_b_ui(self):
        enabled = self.var_comparar_b.get()
        state = "normal" if enabled else "disabled"
        for w in self.frame_b_entries.winfo_children():
            if isinstance(w, (ttk.Entry, ttk.Label)):
                w.configure(state=state)
        self.line_B.set_visible(enabled)
        self.head_B.set_visible(enabled)
        self.actualizar_leyenda()
        self.canvas.draw_idle()

    def cambiar_tema(self, event=None):
        sel = self.cb_tema.get()
        self.current_theme_name = "oscuro" if "Oscuro" in sel else "claro"
        theme = self.THEMES[self.current_theme_name]

        self.aplicar_estilo_ejes()

        self.line_A.set_color(theme["line_a"])
        self.head_A.set_color(theme["head_a"])
        self.line_B.set_color(theme["line_b"])
        self.head_B.set_color(theme["head_b"])
        self.sc_origin.set_color(theme["fp_origin"])
        self.sc_foci.set_color(theme["fp_foci"])
        self.txt_hud.set_color(theme["text_color"])

        self.actualizar_leyenda()
        self.canvas.draw_idle()

    def fijar_vista(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def cargar_preset(self, event=None):
        preset_name = self.cb_preset.get()
        if preset_name not in self.PRESETS:
            return
        p = self.PRESETS[preset_name]

        self.entries["sigma"].delete(0, tk.END)
        self.entries["sigma"].insert(0, str(p["sigma"]))

        self.entries["rho"].delete(0, tk.END)
        self.entries["rho"].insert(0, str(p["rho"]))

        self.entries["beta"].delete(0, tk.END)
        beta_str = "8/3" if abs(p["beta"] - 8/3) < 1e-4 else str(p["beta"])
        self.entries["beta"].insert(0, beta_str)

        self.entries["x0"].delete(0, tk.END)
        self.entries["x0"].insert(0, str(p["x0"]))
        self.entries["y0"].delete(0, tk.END)
        self.entries["y0"].insert(0, str(p["y0"]))
        self.entries["z0"].delete(0, tk.END)
        self.entries["z0"].insert(0, str(p["z0"]))

        self.entries["delta_x"].delete(0, tk.END)
        self.entries["delta_x"].insert(0, str(p["dx"]))

        self.entries["t_max"].delete(0, tk.END)
        self.entries["t_max"].insert(0, str(p["t_max"]))

        self.ejecutar_simulacion()

    def evaluar_expresion(self, valor_str):
        """Permite ingresar fracciones como '8/3' o flotantes estándar."""
        valor_str = valor_str.strip().replace(",", ".")
        if "/" in valor_str:
            num, den = valor_str.split("/", 1)
            return float(num) / float(den)
        return float(valor_str)

    def leer_parametros(self):
        try:
            sigma = self.evaluar_expresion(self.entries["sigma"].get())
            rho = self.evaluar_expresion(self.entries["rho"].get())
            beta = self.evaluar_expresion(self.entries["beta"].get())

            x0 = self.evaluar_expresion(self.entries["x0"].get())
            y0 = self.evaluar_expresion(self.entries["y0"].get())
            z0 = self.evaluar_expresion(self.entries["z0"].get())

            dx = self.evaluar_expresion(self.entries["delta_x"].get())
            dy = self.evaluar_expresion(self.entries["delta_y"].get())
            dz = self.evaluar_expresion(self.entries["delta_z"].get())

            t_max = self.evaluar_expresion(self.entries["t_max"].get())
            dt = self.evaluar_expresion(self.entries["dt"].get())

            if sigma <= 0 or beta <= 0 or rho <= 0:
                raise ValueError("Los parámetros σ, ρ, β deben ser valores positivos.")
            if t_max <= 0 or dt <= 0:
                raise ValueError("El tiempo total y el paso dt deben ser mayores que cero.")
            if dt > t_max:
                raise ValueError("El paso dt no puede ser mayor que el tiempo total.")

            return {
                "sigma": sigma, "rho": rho, "beta": beta,
                "x0": x0, "y0": y0, "z0": z0,
                "dx": dx, "dy": dy, "dz": dz,
                "t_max": t_max, "dt": dt,
                "comparar_b": self.var_comparar_b.get()
            }
        except ValueError as err:
            messagebox.showerror("Error de Parámetros", f"Entrada numérica inválida:\n{err}")
            return None

    def calcular_focos(self, sigma, rho, beta):
        """Calcula los puntos de equilibrio analíticos del sistema de Lorenz."""
        puntos = {"origen": (0.0, 0.0, 0.0), "focos": []}
        if rho > 1:
            d = np.sqrt(beta * (rho - 1))
            z = rho - 1
            puntos["focos"] = [
                (d, d, z),
                (-d, -d, z)
            ]
        return puntos

    def actualizar_focos(self):
        if not self.p_actual:
            return
        if self.var_mostrar_focos.get():
            fp = self.calcular_focos(self.p_actual["sigma"], self.p_actual["rho"], self.p_actual["beta"])
            self.sc_origin.set_data([0], [0])
            self.sc_origin.set_3d_properties([0])
            self.sc_origin.set_visible(True)

            if fp["focos"]:
                fx = [fp["focos"][0][0], fp["focos"][1][0]]
                fy = [fp["focos"][0][1], fp["focos"][1][1]]
                fz = [fp["focos"][0][2], fp["focos"][1][2]]
                self.sc_foci.set_data(fx, fy)
                self.sc_foci.set_3d_properties(fz)
                self.sc_foci.set_visible(True)
            else:
                self.sc_foci.set_visible(False)
        else:
            self.sc_origin.set_visible(False)
            self.sc_foci.set_visible(False)

        self.actualizar_leyenda()
        self.canvas.draw_idle()

    def actualizar_leyenda(self):
        theme = self.THEMES[self.current_theme_name]
        handles = [self.line_A]
        if self.var_comparar_b.get():
            handles.append(self.line_B)
        if self.var_mostrar_focos.get():
            handles.append(self.sc_origin)
            if self.sc_foci.get_visible():
                handles.append(self.sc_foci)

        leg = self.ax.legend(handles=handles, loc="upper right", fontsize=8,
                             facecolor=theme["ax_bg"], edgecolor=theme["grid_color"])
        for text in leg.get_texts():
            text.set_color(theme["text_color"])

    def ejecutar_simulacion(self):
        # Detener animación en curso
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.anim_running = False

        p = self.leer_parametros()
        if not p:
            return
        self.p_actual = p

        self.lbl_telemetria.config(text="Integrando ecuaciones diferenciales (RK45)...")
        self.root.update_idletasks()

        # Vector temporal de evaluación
        num_pasos = int(p["t_max"] / p["dt"])
        self.total_points = max(100, num_pasos)
        self.t_eval = np.linspace(0, p["t_max"], self.total_points)

        # Resolver numéricamente Trayectoria A
        y0_A = [p["x0"], p["y0"], p["z0"]]
        sol_A = solve_ivp(
            lorenz_ode,
            t_span=(0, p["t_max"]),
            y0=y0_A,
            t_eval=self.t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            args=(p["sigma"], p["rho"], p["beta"])
        )

        self.data_A = (sol_A.y[0], sol_A.y[1], sol_A.y[2])

        # Resolver Trayectoria B si está habilitada
        if p["comparar_b"]:
            y0_B = [p["x0"] + p["dx"], p["y0"] + p["dy"], p["z0"] + p["dz"]]
            sol_B = solve_ivp(
                lorenz_ode,
                t_span=(0, p["t_max"]),
                y0=y0_B,
                t_eval=self.t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-8,
                args=(p["sigma"], p["rho"], p["beta"])
            )
            self.data_B = (sol_B.y[0], sol_B.y[1], sol_B.y[2])
            self.line_B.set_visible(True)
            self.head_B.set_visible(True)
        else:
            self.data_B = None
            self.line_B.set_visible(False)
            self.head_B.set_visible(False)

        # Ajustar límites del gráfico 3D
        xA, yA, zA = self.data_A
        all_x, all_y, all_z = list(xA), list(yA), list(zA)
        if self.data_B:
            xB, yB, zB = self.data_B
            all_x.extend(xB)
            all_y.extend(yB)
            all_z.extend(zB)

        pad = 2.0
        self.ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        self.ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        self.ax.set_zlim(min(all_z) - pad, max(all_z) + pad)

        # Título dinámico
        self.ax.set_title(f"Atractor de Lorenz (σ={p['sigma']}, ρ={p['rho']}, β={p['beta']:.3g})",
                          color=self.THEMES[self.current_theme_name]["text_color"], fontsize=11, pad=12)

        self.actualizar_focos()
        self.actualizar_leyenda()

        # Iniciar animación desde el principio
        self.current_idx = 0
        self.slider_tiempo.set(0.0)
        self.anim_running = True
        self.btn_pausa.config(text="⏸ Pausar")

        self.animar_frame()

    def get_speed_multiplier(self):
        sel = self.cb_velocidad.get()
        mapping = {"1x": 1, "2x": 2, "4x": 4, "8x": 8, "16x": 16}
        return mapping.get(sel, 4)

    def animar_frame(self):
        if not self.anim_running or self.data_A is None:
            return

        idx = self.current_idx
        xA, yA, zA = self.data_A

        # Actualizar datos de Trayectoria A
        self.line_A.set_data(xA[:idx + 1], yA[:idx + 1])
        self.line_A.set_3d_properties(zA[:idx + 1])
        self.head_A.set_data([xA[idx]], [yA[idx]])
        self.head_A.set_3d_properties([zA[idx]])

        # Actualizar datos de Trayectoria B
        dist_str = "N/A"
        if self.data_B and self.var_comparar_b.get():
            xB, yB, zB = self.data_B
            self.line_B.set_data(xB[:idx + 1], yB[:idx + 1])
            self.line_B.set_3d_properties(zB[:idx + 1])
            self.head_B.set_data([xB[idx]], [yB[idx]])
            self.head_B.set_3d_properties([zB[idx]])

            dist = np.sqrt((xA[idx] - xB[idx])**2 + (yA[idx] - yB[idx])**2 + (zA[idx] - zB[idx])**2)
            dist_str = f"{dist:.4e}"

        # Actualizar Telemetría HUD
        t_actual = self.t_eval[idx]
        hud_text = (
            f"t = {t_actual:05.2f}s / {self.p_actual['t_max']:.1f}s\n"
            f"Pos A: ({xA[idx]:+06.2f}, {yA[idx]:+06.2f}, {zA[idx]:05.2f})\n"
            f"Separación ||Δr||: {dist_str}"
        )
        self.txt_hud.set_text(hud_text)

        self.lbl_telemetria.config(
            text=f"t: {t_actual:5.2f}s | Frame: {idx+1}/{self.total_points}\n"
                 f"A: x={xA[idx]:.2f}, y={yA[idx]:.2f}, z={zA[idx]:.2f}\n"
                 f"Separación caótica ||Δr||: {dist_str}"
        )

        # Actualizar posición del slider si el usuario no lo está arrastrando
        if not self.user_dragging_slider:
            fraction = idx / max(1, self.total_points - 1)
            self.slider_tiempo.set(fraction)

        # Auto-rotación continua suave de la cámara 3D
        if self.var_auto_rotar.get():
            self.ax.azim = (self.ax.azim + 0.4) % 360

        self.canvas.draw_idle()

        # Avanzar índice de frames según velocidad seleccionada
        step = self.get_speed_multiplier()
        self.current_idx += step

        if self.current_idx >= self.total_points:
            self.current_idx = self.total_points - 1
            self.anim_running = False
            self.btn_pausa.config(text="▶ Reproducir")
            self.lbl_telemetria.config(text=self.lbl_telemetria.cget("text") + " (Completado)")
        else:
            self.anim_job = self.root.after(25, self.animar_frame)

    def on_slider_press(self):
        self.user_dragging_slider = True

    def on_slider_release(self):
        self.user_dragging_slider = False

    def on_slider_change(self, val):
        if self.data_A is None:
            return
        fraction = float(val)
        target_idx = int(fraction * (self.total_points - 1))
        self.current_idx = max(0, min(target_idx, self.total_points - 1))

        # Si estaba pausado, actualizar el frame visualmente al instante
        if not self.anim_running:
            self.actualizar_vista_estatica(self.current_idx)

    def actualizar_vista_estatica(self, idx):
        if self.data_A is None:
            return
        xA, yA, zA = self.data_A
        self.line_A.set_data(xA[:idx + 1], yA[:idx + 1])
        self.line_A.set_3d_properties(zA[:idx + 1])
        self.head_A.set_data([xA[idx]], [yA[idx]])
        self.head_A.set_3d_properties([zA[idx]])

        dist_str = "N/A"
        if self.data_B and self.var_comparar_b.get():
            xB, yB, zB = self.data_B
            self.line_B.set_data(xB[:idx + 1], yB[:idx + 1])
            self.line_B.set_3d_properties(zB[:idx + 1])
            self.head_B.set_data([xB[idx]], [yB[idx]])
            self.head_B.set_3d_properties([zB[idx]])
            dist = np.sqrt((xA[idx] - xB[idx])**2 + (yA[idx] - yB[idx])**2 + (zA[idx] - zB[idx])**2)
            dist_str = f"{dist:.4e}"

        t_actual = self.t_eval[idx]
        self.txt_hud.set_text(
            f"t = {t_actual:05.2f}s / {self.p_actual['t_max']:.1f}s\n"
            f"Pos A: ({xA[idx]:+06.2f}, {yA[idx]:+06.2f}, {zA[idx]:05.2f})\n"
            f"Separación ||Δr||: {dist_str}"
        )
        self.lbl_telemetria.config(
            text=f"t: {t_actual:5.2f}s | Frame: {idx+1}/{self.total_points}\n"
                 f"A: x={xA[idx]:.2f}, y={yA[idx]:.2f}, z={zA[idx]:.2f}\n"
                 f"Separación caótica ||Δr||: {dist_str}"
        )
        self.canvas.draw_idle()

    def toggle_pausa(self):
        if self.data_A is None:
            return
        if self.anim_running:
            self.anim_running = False
            if self.anim_job:
                self.root.after_cancel(self.anim_job)
                self.anim_job = None
            self.btn_pausa.config(text="▶ Reanudar")
        else:
            if self.current_idx >= self.total_points - 1:
                self.current_idx = 0
            self.anim_running = True
            self.btn_pausa.config(text="⏸ Pausar")
            self.animar_frame()

    def reiniciar_animacion(self):
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.current_idx = 0
        self.slider_tiempo.set(0.0)
        if self.data_A is not None:
            self.anim_running = True
            self.btn_pausa.config(text="⏸ Pausar")
            self.animar_frame()

    def dibujar_completo(self):
        """Muestra instantáneamente la trayectoria completa calculada sin esperar la animación."""
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.anim_running = False
        self.btn_pausa.config(text="▶ Reproducir")

        if self.data_A is None:
            return

        self.current_idx = self.total_points - 1
        self.slider_tiempo.set(1.0)
        self.actualizar_vista_estatica(self.current_idx)

    def exportar_imagen(self):
        if self.data_A is None:
            messagebox.showinfo("Aviso", "Primero ejecuta una simulación.")
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png"), ("Documento PDF", "*.pdf"), ("Vector SVG", "*.svg")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
            initialfile="atractor_lorenz_3d.png",
            title="Guardar visualización del Atractor de Lorenz"
        )
        if not ruta:
            return

        try:
            self.fig.savefig(ruta, dpi=300, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
            messagebox.showinfo("Éxito", f"Imagen guardada en alta resolución:\n{ruta}")
        except Exception as ex:
            messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen:\n{ex}")

    def exportar_gif(self):
        if self.data_A is None:
            messagebox.showinfo("Aviso", "Primero ejecuta una simulación para poder exportar.")
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("Animación GIF", "*.gif")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
            initialfile="atractor_lorenz_animado.gif",
            title="Guardar animación 3D como GIF"
        )
        if not ruta:
            return

        # Pausar simulación actual mientras se renderiza
        if self.anim_running:
            self.toggle_pausa()

        self.lbl_telemetria.config(text="Generando GIF animado... Por favor espere.")
        self.root.update_idletasks()

        try:
            theme = self.THEMES[self.current_theme_name]
            fig_gif = Figure(figsize=(7, 6), dpi=90, facecolor=theme["fig_bg"])
            ax_gif = fig_gif.add_subplot(111, projection="3d")
            ax_gif.set_facecolor(theme["ax_bg"])

            # Límites idénticos a los actuales
            ax_gif.set_xlim(self.ax.get_xlim())
            ax_gif.set_ylim(self.ax.get_ylim())
            ax_gif.set_zlim(self.ax.get_zlim())
            ax_gif.set_title(self.ax.get_title(), color=theme["text_color"], fontsize=10)
            ax_gif.tick_params(colors=theme["text_color"], labelsize=7)

            line_gA, = ax_gif.plot([], [], [], lw=1.2, color=theme["line_a"], label="Trayectoria A")
            head_gA, = ax_gif.plot([], [], [], 'o', color=theme["head_a"], markersize=5)

            comparar_b = self.data_B is not None and self.var_comparar_b.get()
            if comparar_b:
                line_gB, = ax_gif.plot([], [], [], lw=1.2, color=theme["line_b"], label="Trayectoria B")
                head_gB, = ax_gif.plot([], [], [], 'o', color=theme["head_b"], markersize=5)

            # Submuestreo para generar ~120 frames fluidos y livianos
            num_frames_gif = 120
            step = max(1, self.total_points // num_frames_gif)
            frames_indices = list(range(0, self.total_points, step))
            if frames_indices[-1] != self.total_points - 1:
                frames_indices.append(self.total_points - 1)

            xA, yA, zA = self.data_A
            if comparar_b:
                xB, yB, zB = self.data_B

            elev_init, azim_init = 20, 45

            def update_gif(f_idx):
                idx = frames_indices[f_idx]
                line_gA.set_data(xA[:idx + 1], yA[:idx + 1])
                line_gA.set_3d_properties(zA[:idx + 1])
                head_gA.set_data([xA[idx]], [yA[idx]])
                head_gA.set_3d_properties([zA[idx]])

                if comparar_b:
                    line_gB.set_data(xB[:idx + 1], yB[:idx + 1])
                    line_gB.set_3d_properties(zB[:idx + 1])
                    head_gB.set_data([xB[idx]], [yB[idx]])
                    head_gB.set_3d_properties([zB[idx]])

                # Rotar la cámara suavemente a lo largo del gif
                azim = (azim_init + f_idx * (360.0 / len(frames_indices))) % 360
                ax_gif.view_init(elev=elev_init, azim=azim)

                return line_gA, head_gA

            anim = animation.FuncAnimation(fig_gif, update_gif, frames=len(frames_indices), blit=False)
            anim.save(ruta, writer="pillow", fps=24)

            self.lbl_telemetria.config(text=f"¡GIF guardado con éxito en {os.path.basename(ruta)}!")
            messagebox.showinfo("Exportación Exitosa", f"El archivo GIF ha sido generado en:\n{ruta}")
        except Exception as ex:
            self.lbl_telemetria.config(text="Error al exportar GIF.")
            messagebox.showerror("Error al guardar GIF", f"Ocurrió un error al generar la animación:\n{ex}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppAtractorLorenz(root)
    root.mainloop()
