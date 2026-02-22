import numpy as np
import cv2

def muestra_imagen(imagen, titulo="Imagen"):
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagen_base = cv2.imread("img/figura3.png", cv2.IMREAD_GRAYSCALE)

if imagen_base is None:
    print("No se pudo cargar la imagen")
    exit()
else:
    print("Imagen cargada correctamente")
    print("Dimensiones:", imagen_base.shape)

muestra_imagen(imagen_base, "Figura original")

class TransformacionesEuclideanas:

    @staticmethod
    def rotacion(angulo):
        return np.array([
            [np.cos(angulo), -np.sin(angulo), 0],
            [np.sin(angulo),  np.cos(angulo), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def traslado(x, y):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

    @staticmethod
    def escala(sx, sy):
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])


# transf. acumulada
h, w = imagen_base.shape
imagen_transformada = np.zeros((h, w), dtype=imagen_base.dtype)

# Centro de la imagen
cx = w // 2
cy = h // 2

T1 = TransformacionesEuclideanas.traslado(-cx, -cy)
R = TransformacionesEuclideanas.rotacion(np.pi / 3)   # 60°
S = TransformacionesEuclideanas.escala(1.5, 1.5)
T2 = TransformacionesEuclideanas.traslado(cx, cy)
T3 = TransformacionesEuclideanas.traslado(30, -60)

T = T3 @ T2 @ R @ S @ T1

print("\nMatriz total empleada:")
print(T)

# Inversa para mapeo
T_inv = np.linalg.inv(T)

# transf. pixelxpixel
for x in range(w):
    for y in range(h):
        p_destino = np.array([x, y, 1])
        p_origen = T_inv @ p_destino

        xo = int(round(p_origen[0]))
        yo = int(round(p_origen[1]))

        if 0 <= xo < w and 0 <= yo < h:
            imagen_transformada[y, x] = imagen_base[yo, xo]

muestra_imagen(imagen_transformada, "Figura transformada")