import tkinter as tk
from tkinter import messagebox, ttk
import pymysql
import heapq
import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
try:
    model = tf.keras.models.load_model('modelo_entrenado_rutas.keras')
    messagebox.showinfo("Modelo Cargado", "Modelo cargado correctamente")
except Exception as e:
    messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")

# --- Algoritmo de Dijkstra ---
def dijkstra(graph, start, end):
    priority_queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end:
            return current_distance

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')

# --- Base de datos MySQL ---
def conectar_db():
    try:
        db = pymysql.connect(host="localhost", user="root", passwd="", db="bddprediccionesrutas")
        return db
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo conectar a la base de datos:\n{e}")
        return None

def init_db():
    db = conectar_db()
    if db is None:
        return
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resultados (
        id INT AUTO_INCREMENT PRIMARY KEY,
        origen VARCHAR(255),
        destino VARCHAR(255),
        distancia INT,
        opciones INT
    )
    """)
    db.commit()
    db.close()

def guardar_resultado(origen, destino, distancia, opciones):
    db = conectar_db()
    if db is None:
        return
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO resultados (origen, destino, distancia, opciones) VALUES (%s, %s, %s, %s)", 
            (origen, destino, distancia, opciones)
        )
        db.commit()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo guardar en la base de datos:\n{e}")
    finally:
        db.close()

def listar_resultados():
    db = conectar_db()
    if db is None:
        return

    cursor = db.cursor()
    try:
        cursor.execute("SELECT id, origen, destino, distancia, opciones FROM resultados")
        resultados = cursor.fetchall()
        
        ventana_listado = tk.Toplevel(window)
        ventana_listado.title("Resultados Guardados")
        ventana_listado.geometry("500x400")

        tree = ttk.Treeview(ventana_listado, columns=("ID", "Origen", "Destino", "Distancia", "Opciones"), show="headings")
        tree.heading("ID", text="ID")
        tree.heading("Origen", text="Origen")
        tree.heading("Destino", text="Destino")
        tree.heading("Distancia", text="Distancia")
        tree.heading("Opciones", text="Opciones")
        tree.column("ID", width=50)
        tree.column("Origen", width=100)
        tree.column("Destino", width=100)
        tree.column("Distancia", width=100)
        tree.column("Opciones", width=100)

        for row in resultados:
            tree.insert("", tk.END, values=row)

        tree.pack(expand=True, fill=tk.BOTH)

        def eliminar_seleccion():
            seleccionado = tree.selection()
            if not seleccionado:
                messagebox.showerror("Error", "Selecciona un registro para eliminar.")
                return

            id_seleccionado = tree.item(seleccionado[0])['values'][0]
            eliminar_resultado(id_seleccionado)
            tree.delete(seleccionado[0])
            messagebox.showinfo("Eliminado", "El registro ha sido eliminado exitosamente.")

        btn_eliminar = tk.Button(ventana_listado, text="Eliminar Seleccionado", command=eliminar_seleccion)
        btn_eliminar.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudieron listar los resultados:\n{e}")
    finally:
        db.close()

def eliminar_resultado(id):
    db = conectar_db()
    if db is None:
        return
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM resultados WHERE id = %s", (id,))
        db.commit()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar el registro:\n{e}")
    finally:
        db.close()

# --- Predicción de distancia usando el modelo de IA ---
def predecir_distancia(origen, destino, opciones_a, opciones_b):
    # Convertir las entradas (origen, destino) en IDs numéricos
    origen_id = ord(origen) - ord('A') + 1  # Simple conversión a ID
    destino_id = ord(destino) - ord('A') + 1
    total_opciones = opciones_a + opciones_b  # Sumar las opciones

    # Preparar los datos para la predicción
    input_data = np.array([[origen_id, destino_id, total_opciones]])

    # Predecir la distancia usando el modelo entrenado
    try:
        predicted_distance = model.predict(input_data)
        return predicted_distance[0][0]
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo hacer la predicción:\n{e}")
        return None

# --- Interfaz gráfica ---
def calcular_ruta():
    origen = entry_origen.get().strip()
    destino = entry_destino.get().strip()
    opciones_a = int(entry_ruta_a.get().strip())
    opciones_b = int(entry_ruta_b.get().strip())

    if origen not in graph or destino not in graph:
        messagebox.showerror("Error", "Nodos no válidos en el grafo")
        return
    
    distancia_dijkstra = dijkstra(graph, origen, destino)
    distancia_predicha = predecir_distancia(origen, destino, opciones_a, opciones_b)
    
    if distancia_dijkstra == float('inf'):
        messagebox.showerror("Error", "No hay ruta disponible")
    elif distancia_predicha is None:
        messagebox.showerror("Error", "Hubo un problema al predecir la distancia")
    else:
        total_opciones = opciones_a + opciones_b
        guardar_resultado(origen, destino, distancia_predicha, total_opciones)
        messagebox.showinfo(
            "Resultado",
            f"Distancia más corta de {origen} a {destino} (Dijkstra): {distancia_dijkstra}\n"
            f"Distancia (suma) predicha por IA: {distancia_predicha}\n"
            f"Total de opciones posibles: {total_opciones}"
        )

# Grafo de ejemplo
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8},
    'D': {'B': 5, 'C': 8}
}

# Inicializar base de datos
init_db()

# Crear ventana
window = tk.Tk()
window.title("Optimización de Rutas")
window.geometry("400x400")

# Etiquetas y campos de entrada
label_origen = tk.Label(window, text="Nodo de origen:")
label_origen.pack(pady=5)
entry_origen = tk.Entry(window)
entry_origen.pack(pady=5)

label_destino = tk.Label(window, text="Nodo de destino:")
label_destino.pack(pady=5)
entry_destino = tk.Entry(window)
entry_destino.pack(pady=5)

label_ruta_a = tk.Label(window, text="Opciones Ruta A:")
label_ruta_a.pack(pady=5)
entry_ruta_a = tk.Entry(window)
entry_ruta_a.pack(pady=5)

label_ruta_b = tk.Label(window, text="Opciones Ruta B:")
label_ruta_b.pack(pady=5)
entry_ruta_b = tk.Entry(window)
entry_ruta_b.pack(pady=5)

# Botones
btn_calcular = tk.Button(window, text="Calcular Ruta", command=calcular_ruta)
btn_calcular.pack(pady=10)

btn_ver_resultados = tk.Button(window, text="Ver Resultados Guardados", command=listar_resultados)
btn_ver_resultados.pack(pady=10)

# Ejecutar ventana
window.mainloop()
