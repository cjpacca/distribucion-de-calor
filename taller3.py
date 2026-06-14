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
# CONSTRUCCIÓN DEL VECTOR DE TÉRMINOS INDEPENDIENTES (b)
# ============================================================
print("\nIniciando construcción del vector b...")
b = np.zeros(N_total)

centro_i = n // 2
centro_j = n // 2
k_centro = centro_i * n + centro_j
b[k_centro] += f_intensidad * (h**2)

# Borde Superior (Fila i=0 -> Índices del 0 al n-1)
b[0 : n] += T_superior

# Borde Inferior (Fila i=n-1 -> Índices desde (n-1)*n hasta el final)
b[(n - 1) * n : N_total] += T_inferior

# Borde Izquierdo (Columna j=0 -> Índices de 0 a N_total saltando de n en n)
b[0 : N_total : n] += T_izquierda

# Borde Derecha (Columna j=n-1 -> Índices de n-1 a N_total saltando de n en n)
b[n - 1 : N_total : n] += T_derecha

# ============================================================
# SOLUCIÓN DEL SISTEMA POR CHOLESKY
# ============================================================

print("\nResolviendo el sistema de ecuaciones...")

# la.cholesky con lower=True calcula la descomposición exacta de Cholesky
print("1. Calculando la matriz triangular inferior L...")
L = la.cholesky(A, lower=True)

print("2. Resolviendo L * y = b (Sustitución hacia adelante)...")
y = la.solve_triangular(L, b, lower=True)

print("3. Resolviendo L^T * u = y (Sustitución hacia atrás)...")
u_lineal = la.solve_triangular(L, y, lower=True, trans=1)  # trans=1 es para usar L^T

# Se redimensiona el vector a la malla
U_interno = u_lineal.reshape((n, n))

print(f"Forma de la matriz de Cholesky L: {L.shape}")
print(f"Temperatura en el centro de la placa: {U_interno[centro_i, centro_j]:.6f}°C")
print("-" * 40)

# ============================================================
# VISUALIZACIÓN
# ============================================================

# Patrón de dispersión
plt.figure(figsize=(6, 6))
plt.spy(A, markersize=1, color="black")
plt.title("Estructura de Bandas de la Matriz A")
plt.xlabel("Columnas (j)")
plt.ylabel("Filas (i)")
plt.grid(True, alpha=0.3)
plt.show()

# Ahora debemos reconstruir la placa, necesitamos verla desde el borde exterior para que esté completa
# Creamos la matriz extendida de 22x22 para incluir los bordes
U_completa = np.zeros((n + 2, n + 2))
U_completa[1 : n + 1, 1 : n + 1] = U_interno

# Aplicamos las temperaturas de los bordes
U_completa[0, :] = T_superior
U_completa[-1, :] = T_inferior
U_completa[:, 0] = T_izquierda
U_completa[:, -1] = T_derecha 

# Mapa de calor con isotermas
plt.figure(figsize=(7, 6))
# imshow dibuja la matriz en 2D. origin="upper" hace que la fila 0 quede arriba.
plt.imshow(U_completa, cmap="jet", origin="upper")
plt.colorbar(label="Temperatura (°C)")

print("\n")
# Añadimos las curvas de nivel (isotermas)
# Usamos un número fijo de niveles para ver la gradación
plt.contour(U_completa, colors="white", levels=12, alpha=0.6)
plt.title("Distribución Estacionaria de Temperatura")
plt.xlabel("Índice X")
plt.ylabel("Índice Y")
plt.show()