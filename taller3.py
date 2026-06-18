import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

# Definimos los parámetros que representan el dominio del problema

x_min = 0      # Límite izquierdo
x_max = 1      # Límite derecho
y_min = 0      # Límite inferior
y_max = 1      # Límite superior

n = 20          # Número de nodos internos por lado (no incluye los de la frontera)
N_total = n * n # Total de nodos internos (incógnitas) en la malla

h = 1 / (n + 1) # Espaciado entre nodos osea el tamaño de la malla

print("PARÁMETROS")

print(f"Placa: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
print(f"Nodos internos por lado: {n}")
print(f"Total de nodos internos: {N_total}")
print(f"Espaciado h = 1/(n+1) = {h:.6f}")
# Creamos la malla de coordenadas

x_coords = []
y_coords = []

for i in range(1, n + 1):
    x = x_min + i * h
    x_coords.append(x)

for j in range(1, n + 1):
    y = y_min + j * h
    y_coords.append(y)

# Convertimoslas listas a arrays de numpy para las operaciones vectoriales
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# Ahora si creamos la malla y obtenemos 2 matrices 2D
X, Y = np.meshgrid(x_coords, y_coords)

print(f"Malla de coordenadas: {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords)}")
print("-" * 40)
def indice_2d_a_1d(i, j, n):
    k = i * n + j # la fórmula está en row-major
    return k

def indice_1d_a_2d(k, n):
    i = k // n
    j = k % n
    return i, j

# ============================================================
# CONSTRUCCIÓN DE LA MATRIZ DEL SISTEMA (A)
# ============================================================

A = np.zeros((N_total, N_total))
print("\nIniciando construcción de la matriz A...")
print(f"Forma de A: {A.shape}")
print(f"Número de elementos: {A.size}")

# Insertamos los coeficientes de la ecuación de Poisson discretizada

for i in range(n):
    for j in range(n):

        k = indice_2d_a_1d(i, j, n)
        # Coeficiente del nodo central, este viene de la ecuación de Poisson al desarrollar la expresión
        A[k, k] = 4 # este va en la diagonal, la temperatura depende principalmente de su propia temperatura y sus vecinos también la afectan pero negativamente

        # Vecino de arriba (i-1, j)
        if i > 0:
            k_arriba = indice_2d_a_1d(i - 1, j, n)
            A[k, k_arriba] = -1 # La temperatura del nodo (i, j) es menor si la temperatura de arriba es mayor

        # Vecino de abajo (i+1, j)
        if i < n - 1:
            k_abajo = indice_2d_a_1d(i + 1, j, n)
            A[k, k_abajo] = -1 # La temperatura del nodo (i, j) es menor si la temperatura de abajo es mayor

        # Vecino de la izquierda (i, j-1)
        if j > 0:
            k_izquierda = indice_2d_a_1d(i, j - 1, n)
            A[k, k_izquierda] = -1

        # Vecino de la derecha (i, j+1)
        if j < n - 1:
            k_derecha = indice_2d_a_1d(i, j + 1, n)
            A[k, k_derecha] = -1

# Definimos las condiciones de Dirichlet
# Sus valores se aplicarán directamente en la construcción del vector b

T_superior = 100
T_inferior = 0 
T_izquierda = 50
T_derecha = 50
f_intensidad = 10000
print("Constantes de la frontera:")
print("-" * 40)
print(f"  Superior: {T_superior}°C | Inferior: {T_inferior}°C")
print(f"  Izquierda: {T_izquierda}°C | Derecha: {T_derecha}°C")
print("-" * 40)

# ============================================================
# CONSTRUCCIÓN DE LOS VECTORES DE TÉRMINOS INDEPENDIENTES (b)
# ============================================================
print("\nIniciando construcción de los vectores b...")

# 1. Creamos el vector base SOLO con las condiciones de frontera
b_sin_fuente = np.zeros(N_total)

# Borde Superior e Inferior
b_sin_fuente[0 : n] += T_superior
b_sin_fuente[(n - 1) * n : N_total] += T_inferior

# Borde Izquierdo y Derecho
b_sin_fuente[0 : N_total : n] += T_izquierda
b_sin_fuente[n - 1 : N_total : n] += T_derecha

# 2. Creamos una copia para el escenario CON fuente y le inyectamos el calor
b_con_fuente = np.copy(b_sin_fuente)
centro_i = n // 2
centro_j = n // 2
k_centro = centro_i * n + centro_j

# Incrementamos la intensidad temporalmente para que el pico sea bien visible en la gráfica
f_intensidad = 100000 
b_con_fuente[k_centro] += f_intensidad * (h**2)

# ============================================================
# SOLUCIÓN DEL SISTEMA POR CHOLESKY (AMBOS ESCENARIOS)
# ============================================================
print("\nResolviendo los sistemas de ecuaciones...")
print("1. Calculando la matriz triangular inferior L...")
L = la.cholesky(A, lower=True)

print("2. Resolviendo escenario SIN fuente de calor...")
y_sin = la.solve_triangular(L, b_sin_fuente, lower=True)
u_lineal_sin = la.solve_triangular(L, y_sin, lower=True, trans=1)
U_interno_sin = u_lineal_sin.reshape((n, n))

print("3. Resolviendo escenario CON fuente de calor...")
y_con = la.solve_triangular(L, b_con_fuente, lower=True)
u_lineal_con = la.solve_triangular(L, y_con, lower=True, trans=1)
U_interno_con = u_lineal_con.reshape((n, n))

# ============================================================
# RECONSTRUCCIÓN DE LAS MATRICES COMPLETAS (INCLUYENDO BORDES)
# ============================================================
U_completa_sin = np.zeros((n + 2, n + 2))
U_completa_con = np.zeros((n + 2, n + 2))

# Insertar nodos internos
U_completa_sin[1 : n + 1, 1 : n + 1] = U_interno_sin
U_completa_con[1 : n + 1, 1 : n + 1] = U_interno_con

# Aplicar fronteras a ambas matrices
for U in (U_completa_sin, U_completa_con):
    U[0, :] = T_superior
    U[-1, :] = T_inferior
    U[:, 0] = T_izquierda
    U[:, -1] = T_derecha 

# ============================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================

# 1. Estructura de la Matriz A
plt.figure(figsize=(5, 5))
plt.spy(A, markersize=1, color="black")
plt.title("Estructura de Bandas de la Matriz A")
plt.xlabel("Columnas (j)")
plt.ylabel("Filas (i)")
plt.grid(True, alpha=0.3)
plt.show()

# Definimos los límites físicos para los heatmaps
limites_grafico = [x_min, x_max, y_max, y_min]

# 2. Heatmap: SIN Fuente de calor
plt.figure(figsize=(7, 6))
plt.imshow(U_completa_sin, cmap="jet", origin="upper", extent=limites_grafico)
plt.colorbar(label="Temperatura (°C)")
plt.contour(U_completa_sin, colors="white", levels=12, alpha=0.6, extent=limites_grafico)
plt.title("Distribución de Temperatura (Sin Fuente)")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.show()

# 3. Heatmap: CON Fuente de calor
plt.figure(figsize=(7, 6))
plt.imshow(U_completa_con, cmap="jet", origin="upper", extent=limites_grafico)
plt.colorbar(label="Temperatura (°C)")
plt.contour(U_completa_con, colors="white", levels=12, alpha=0.6, extent=limites_grafico)
plt.title("Distribución de Temperatura (Con Fuente)")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.show()

# 4. Perfil Transversal u(x, 0.5)
# Creamos un vector de coordenadas X que incluye los bordes para la gráfica 2D
x_coords_completo = np.linspace(x_min, x_max, n + 2)

# Extraemos la fila central (índice centro_i + 1 debido al borde extra)
perfil_sin = U_completa_sin[centro_i + 1, :]
perfil_con = U_completa_con[centro_i + 1, :]

plt.figure(figsize=(8, 5))
plt.plot(x_coords_completo, perfil_sin, label="Sin fuente", linestyle="--", color="blue")
plt.plot(x_coords_completo, perfil_con, label="Con fuente puntual", color="red", linewidth=2)
plt.title("Perfil Transversal de Temperatura en y = 0.5")
plt.xlabel("Posición X")
plt.ylabel("Temperatura u(x, 0.5) [°C]")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()