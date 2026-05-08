"""
Script para verificar cómo se aplican los kernels en las binarizaciones
"""
import numpy as np
import cv2
import sys
sys.path.insert(0, 'src')

from transforms.Binarization.BinarizationTransforms import Otsu_binarization, Spline_binarization, KERNELS, MORPHS

# Crear imagen de prueba
test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

print("=" * 70)
print("VERIFICACIÓN DE KERNELS EN BINARIZACIONES")
print("=" * 70)

# 1. Verificar definición de kernels
print("\n1. KERNELS DEFINIDOS:")
for key, kernel in KERNELS.items():
    print(f"   Kernel {key}: Shape {kernel.shape}, dtype {kernel.dtype}")
    print(f"      {kernel}")
    
# 2. Verificar morphs
print("\n2. OPERACIONES MORFOLÓGICAS:")
for key, morph in MORPHS.items():
    print(f"   Morph {key}: {morph} (CV2 constant value: {morph})")

# 3. Probar Otsu con diferentes kernels
print("\n3. PROBANDO OTSU BINARIZATION CON DIFERENTES KERNELS:")
for kernel_id in range(1, 5):
    try:
        otsu = Otsu_binarization(morph=1, kernel=kernel_id)
        result = otsu(test_img)
        print(f"   ✓ Kernel {kernel_id} (shape {KERNELS[kernel_id].shape}): SUCCESS")
        print(f"      Output shape: {result.shape}, dtype: {result.dtype}")
    except Exception as e:
        print(f"   ✗ Kernel {kernel_id}: ERROR - {e}")

# 4. Verificar parámetros reales aplicados
print("\n4. VERIFICANDO APLICACIÓN REAL DEL KERNEL:")
otsu_test = Otsu_binarization(morph=1, kernel=2)
print(f"   Otsu instance kernel: {otsu_test.kernel}")
print(f"   Otsu instance morph: {otsu_test.morph}")

# 5. Aplicar manualmente para ver diferencia
print("\n5. COMPARANDO CON Y SIN MORFOLOGÍA:")
_, binary = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"   Sin morfología - Píxeles blancos: {np.sum(binary > 0)}")

kernel = KERNELS[2]  # 3x3
binary_morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
print(f"   Con MORPH_CLOSE 3x3 - Píxeles blancos: {np.sum(binary_morph > 0)}")

# 6. Probar las transformaciones reales
print("\n6. PROBANDO TRANSFORMACIONES REALES:")
otsu1 = Otsu_binarization(morph=1, kernel=2)
result1 = otsu1(test_img)
print(f"   Otsu(morph=1, kernel=2) - Output shape: {result1.shape}")

otsu2 = Otsu_binarization(morph=2, kernel=3)
result2 = otsu2(test_img)
print(f"   Otsu(morph=2, kernel=3) - Output shape: {result2.shape}")

otsu3 = Otsu_binarization(morph=None, kernel=None)
result3 = otsu3(test_img)
print(f"   Otsu(morph=None, kernel=None) - Output shape: {result3.shape}")

print("\n" + "=" * 70)
print("RESUMEN:")
print("=" * 70)
print("""
⚠️ PROBLEMAS ENCONTRADOS:

1. KERNELS PARES (4x4, 5x5):
   - cv2.morphologyEx() espera kernels con ANCHOR point definido
   - Kernels pares (4x4, 5x5) NO tienen anchor central único
   - CV2 puede no aplicarlos correctamente o usar comportamiento no esperado

2. KERNELS SIMPLES:
   - Todos los kernels son np.ones() rectangulares
   - No hay kernels circulares, elípticos o cruzados
   - Esto limita la variedad de operaciones morfológicas

3. RECOMENDACIONES:
   ✓ Usar solo kernels IMPARES (3x3, 5x5, 7x7)
   ✓ Usar cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) para kernels elípticos
   ✓ Usar cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) para kernels en cruz
   ✓ Documentar qué kernel se usa en cada experimento
""")
