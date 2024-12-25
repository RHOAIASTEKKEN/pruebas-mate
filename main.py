import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import math
import pandas as pd

def calcular_derivadas_parciales(funcion_str):
    try:
        x_sym, y_sym = sp.symbols('x y')
        funcion_sym = sp.sympify(funcion_str, locals={'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 
                                                      'exp': sp.exp, 'log': sp.log, 
                                                      'sqrt': sp.sqrt, 'pi': sp.pi})
        df_dx = sp.diff(funcion_sym, x_sym)
        df_dy = sp.diff(funcion_sym, y_sym)
        return str(df_dx), str(df_dy)
    except Exception as e:
        return f"Error al calcular derivadas: {e}", f"Error al calcular derivadas: {e}"

def gradiente(func, x, y, h=1e-5):
    df_dx = (func(x + h, y) - func(x, y)) / h
    df_dy = (func(x, y + h) - func(x, y)) / h
    return df_dx, df_dy

def calcular_camino_gradiente(funcion, x0, y0, pasos=50, alpha=0.1):
    x_camino, y_camino, z_camino = [x0], [y0], [funcion(x0, y0)]
    
    for _ in range(pasos):
        grad_x, grad_y = gradiente(funcion, x_camino[-1], y_camino[-1])
        nuevo_x = x_camino[-1] + alpha * grad_x
        nuevo_y = y_camino[-1] + alpha * grad_y
        nuevo_z = funcion(nuevo_x, nuevo_y)

        x_camino.append(nuevo_x)
        y_camino.append(nuevo_y)
        z_camino.append(nuevo_z)

    return x_camino, y_camino, z_camino

def actualizar_tabla(tree, x_camino, y_camino, z_camino):
    for item in tree.get_children():
        tree.delete(item)
    
    for i in range(len(x_camino)):
        tree.insert('', 'end', values=(
            i,
            f"{x_camino[i]:.4f}",
            f"{y_camino[i]:.4f}",
            f"{z_camino[i]:.4f}"
        ))

def crear_esfera(x, y, z, color='red'):
    # Crear los puntos para una esfera con radio fijo de 0.5
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    radio = 0.5  # Radio fijo de la esfera
    
    x_esfera = x + radio * np.outer(np.cos(u), np.sin(v))
    y_esfera = y + radio * np.outer(np.sin(u), np.sin(v))
    z_esfera = z + radio * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return go.Surface(
        x=x_esfera,
        y=y_esfera,
        z=z_esfera,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.8
    )

def graficar_funcion():
    funcion_str = entrada_funcion.get()
    try:
        punto_x = float(entrada_punto_x.get())
        punto_y = float(entrada_punto_y.get())

        math_dict = {
            'sin': sp.sin, 
            'cos': sp.cos, 
            'tan': sp.tan, 
            'exp': sp.exp, 
            'log': sp.log, 
            'sqrt': sp.sqrt, 
            'pi': sp.pi,
        }

        x_sym, y_sym = sp.symbols('x y')
        funcion_sym = sp.sympify(funcion_str, locals=math_dict)
        funcion = sp.lambdify((x_sym, y_sym), funcion_sym, 
                               modules=[{'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 
                                         'exp': np.exp, 'log': np.log, 
                                         'sqrt': np.sqrt, 'pi': np.pi}, 'numpy'])

        df_dx_str, df_dy_str = calcular_derivadas_parciales(funcion_str)
        etiqueta_dx.config(text=f"∂f/∂x: {df_dx_str}")
        etiqueta_dy.config(text=f"∂f/∂y: {df_dy_str}")

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = funcion(X, Y)

        # Calcular el camino del gradiente
        pasos = 50
        alpha = 0.2
        x_camino, y_camino, z_camino = calcular_camino_gradiente(funcion, punto_x, punto_y, pasos, alpha)

        # Actualizar la tabla con los nuevos datos
        actualizar_tabla(tabla_puntos, x_camino, y_camino, z_camino)

        # Crear frames para la animación
        frames = []
        for i in range(len(x_camino)):
            frame = go.Frame(
                data=[
                    # Superficie base
                    go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7, showscale=False),
                    # Camino del gradiente hasta el punto actual
                    go.Scatter3d(
                        x=x_camino[:i+1],
                        y=y_camino[:i+1],
                        z=z_camino[:i+1],
                        mode='lines',
                        line=dict(color='blue', width=3),
                        showlegend=False
                    ),
                    # Esfera en la posición actual
                    crear_esfera(x_camino[i], y_camino[i], z_camino[i])
                ],
                name=f'frame{i}'
            )
            frames.append(frame)

        # Crear la figura con la animación
        fig = go.Figure(
            data=[
                # Superficie inicial
                go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7, showscale=False),
                # Camino inicial (vacío)
                go.Scatter3d(x=[x_camino[0]], y=[y_camino[0]], z=[z_camino[0]], 
                           mode='lines', line=dict(color='blue', width=3)),
                # Esfera inicial
                crear_esfera(x_camino[0], y_camino[0], z_camino[0])
            ],
            frames=frames
        )

        # Configurar el layout con controles de animación
        fig.update_layout(
            title=f'Visualización 3D: {funcion_str}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                # Configurar aspectos de la escena para mejor visualización
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'  # Mantener proporciones reales
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Paso: '},
                'steps': [
                    {
                        'method': 'animate',
                        'label': str(i),
                        'args': [[f'frame{i}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                    for i in range(len(frames))
                ]
            }]
        )

        fig.show()

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo graficar la función: {e}")
        etiqueta_dx.config(text="∂f/∂x: ")
        etiqueta_dy.config(text="∂f/∂y: ")

root = tk.Tk()
root.title("Graficador de Derivadas Parciales 3D")
root.geometry("800x600")

# Frame principal
frame_principal = ttk.Frame(root)
frame_principal.pack(expand=True, fill='both', padx=10, pady=10)

# Frame izquierdo para controles
frame_controles = ttk.Frame(frame_principal)
frame_controles.pack(side='left', fill='y', padx=(0, 10))

ttk.Label(frame_controles, text="Ingresar función de x e y:").pack(pady=10)
entrada_funcion = ttk.Entry(frame_controles, width=50)
entrada_funcion.pack(pady=5)
entrada_funcion.insert(0, "exp(-x**2-y**2)")

ttk.Label(frame_controles, text="Punto X:").pack(pady=5)
entrada_punto_x = ttk.Entry(frame_controles, width=20)
entrada_punto_x.pack(pady=5)
entrada_punto_x.insert(0, "-1")

ttk.Label(frame_controles, text="Punto Y:").pack(pady=5)
entrada_punto_y = ttk.Entry(frame_controles, width=20)
entrada_punto_y.pack(pady=5)
entrada_punto_y.insert(0, "3")

boton_graficar = ttk.Button(frame_controles, text="Graficar y Calcular Derivadas", command=graficar_funcion)
boton_graficar.pack(pady=10)

etiqueta_dx = ttk.Label(frame_controles, text="∂f/∂x: ", wraplength=380)
etiqueta_dx.pack(pady=5)

etiqueta_dy = ttk.Label(frame_controles, text="∂f/∂y: ", wraplength=380)
etiqueta_dy.pack(pady=5)

# Frame derecho para la tabla
frame_tabla = ttk.Frame(frame_principal)
frame_tabla.pack(side='right', fill='both', expand=True)

# Crear tabla
ttk.Label(frame_tabla, text="Puntos del Camino del Gradiente").pack(pady=5)
tabla_puntos = ttk.Treeview(frame_tabla, columns=('Paso', 'X', 'Y', 'Z'), show='headings')
tabla_puntos.heading('Paso', text='Paso')
tabla_puntos.heading('X', text='X')
tabla_puntos.heading('Y', text='Y')
tabla_puntos.heading('Z', text='Z')

tabla_puntos.column('Paso', width=50)
tabla_puntos.column('X', width=100)
tabla_puntos.column('Y', width=100)
tabla_puntos.column('Z', width=100)

scrollbar = ttk.Scrollbar(frame_tabla, orient='vertical', command=tabla_puntos.yview)
tabla_puntos.configure(yscrollcommand=scrollbar.set)

tabla_puntos.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

root.mainloop()