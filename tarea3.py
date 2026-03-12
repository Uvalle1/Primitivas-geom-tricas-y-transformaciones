import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ajuste canal Y
def ajustar_yuv(imagen, brillo=0, contraste=1.0, gamma=1.0):

    imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(imagen_yuv)

    # Brillo y contraste
    Y = contraste * Y + brillo
    Y = np.clip(Y, 0, 255).astype(np.uint8)

    # Corrección gamma
    Y = np.array(255 * (Y / 255) ** gamma, dtype='uint8')

    imagen_mod = cv2.merge([Y, U, V])
    imagen_mod = cv2.cvtColor(imagen_mod, cv2.COLOR_YUV2BGR)

    return imagen_mod


# FUNCION ECUALIZACION
def ecualizar_yuv(imagen):

    imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(imagen_yuv)

    Y_eq = cv2.equalizeHist(Y)

    imagen_eq = cv2.merge([Y_eq, U, V])
    imagen_eq = cv2.cvtColor(imagen_eq, cv2.COLOR_YUV2BGR)

    return imagen_eq


# Cargar img
ruta = "img/izquierda1.jpeg"
imagen = cv2.imread(ruta)

if imagen is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Parametros
brillo = 60
contraste = 3
gamma = 9


# Procesamiento
imagen_ajustada = ajustar_yuv(imagen, brillo, contraste, gamma)
imagen_eq = ecualizar_yuv(imagen)


# Convertir a RGB para matplot
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_ajustada_rgb = cv2.cvtColor(imagen_ajustada, cv2.COLOR_BGR2RGB)
imagen_eq_rgb = cv2.cvtColor(imagen_eq, cv2.COLOR_BGR2RGB)


# Obtener canal Y
def obtener_Y(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]

Y_original = obtener_Y(imagen)
Y_ajustada = obtener_Y(imagen_ajustada)
Y_eq = obtener_Y(imagen_eq)


# Mostrar todo
plt.figure(figsize=(15, 10))

# Imágenes
plt.subplot(3, 3, 1)
plt.imshow(imagen_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(imagen_ajustada_rgb)
plt.title("Ajustada")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(imagen_eq_rgb)
plt.title("Ecualizada")
plt.axis("off")

# Histogramas
plt.subplot(3, 3, 4)
plt.hist(Y_original.ravel(), 256, [0, 256])
plt.title("Histograma Original")

plt.subplot(3, 3, 5)
plt.hist(Y_ajustada.ravel(), 256, [0, 256])
plt.title("Histograma Ajustado")

plt.subplot(3, 3, 6)
plt.hist(Y_eq.ravel(), 256, [0, 256])
plt.title("Histograma Ecualizado")

# Mostrar parámetros
texto = f"Brillo = {brillo} | Contraste = {contraste} | Gamma = {gamma}"
plt.figtext(0.5, 0.02, texto,
            ha="center",
            fontsize=12,
            bbox={"facecolor": "lightgray", "alpha": 0.6})

plt.tight_layout()
plt.show()


# Imprimir valores
print("\n===== PARÁMETROS UTILIZADOS =====")
print(f"Brillo: {brillo}")
print(f"Contraste: {contraste}")
print(f"Gamma: {gamma}")
print("=================================\n")