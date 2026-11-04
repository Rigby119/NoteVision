# Evaluación

Este documento describe el protocolo de evaluación de NoteVision y los resultados del modelo inicial (**EXP-00**) sobre el conjunto de validación. La arquitectura y el entrenamiento están en [`architecture.md`](architecture.md); los límites del trabajo, en [`limitations.md`](limitations.md).

> **Estado:** EXP-00 evaluado sobre validación completa. El conjunto de prueba **no se ha utilizado**: se evaluará una sola vez, al seleccionar el modelo final. La comparación con EXP-01 se añadirá cuando termine su entrenamiento.

---

## 1. Protocolo

| Aspecto | Decisión |
|---|---|
| Particiones | 80/10/10 por grupo, semilla 42: entrenamiento 70 138, validación 8 772, prueba 8 768 |
| Selección del mejor modelo | Menor CER en una submuestra fija de 2 000 imágenes de validación (`artifacts/val_subset_ids.csv`), común a todos los experimentos |
| Evaluación final | Validación completa (8 772 imágenes) con `best.weights.h5` y decodificación *greedy* CTC, en `notebooks/04_evaluation.ipynb` |
| Conjunto de prueba | Reservado; una sola evaluación al elegir el modelo final |
| Protocolo común | Mismas particiones, métricas, calendario (8 épocas) y criterio de mejor modelo en todos los experimentos |

## 2. Métricas

- **CER (principal):** suma de las distancias de edición (inserciones, omisiones y sustituciones) entre la secuencia predicha y la de referencia, dividida entre el total de tokens de referencia.
- **Exactitud de secuencia:** proporción de ejemplos cuya secuencia completa coincide con la referencia.
- **CER de forma (`cer_shape`):** el mismo cálculo comparando solo la familia del símbolo (nota, silencio, clave, etc.) e ignorando la posición vertical. Separa los errores de símbolo de los de altura.
- **Recall por familia:** aciertos entre apariciones, a partir del alineamiento entre referencia y predicción.
- **Línea base:** predictores triviales sin imagen (la secuencia más frecuente y la moda por posición), con CER ≈ 0,91 en validación.
- **Métrica secundaria prevista (CER sin barra final):** el CER ignorando una `barline-L1` final en referencia y predicción, para neutralizar la inconsistencia de etiquetas descrita en la sección 7.

## 3. Verificaciones previas al entrenamiento completo

Antes de entrenar EXP-00 se comprobó que el pipeline aprende:

- **Sobreajuste de 64 muestras:** tras 400 épocas, CER 0,0093 y exactitud de secuencia 0,938 (con la convención anterior del blanco CTC; no se repitió).
- **Prueba corta con datos reales** (6 000 de entrenamiento, 1 000 de validación, 6 épocas de 146 pasos): el CER de validación pasó de 0,9809 a 0,9362, 0,9394, 0,9786, 0,7072 y 0,2957. En la época 6, la exactitud de secuencia fue 0,013 y el CER de forma 0,2267. Las primeras épocas muestran la meseta característica de CTC, que termina alrededor del paso 600. La velocidad fue de 0,98 a 1,65 s por paso (30 a 51 min por época completa, estimado).

## 4. Entrenamiento de EXP-00

Evolución sobre la submuestra de validación de 2 000 imágenes (3,94 h en total, unos 29,5 min por época):

| Época | Pérdida | CER | Exactitud de secuencia | CER de forma |
|---|---|---|---|---|
| 1 | 58,64 | 0,0317 | 0,592 | 0,0257 |
| 2 | 2,85 | 0,0156 | 0,760 | 0,0125 |
| 3 | 1,68 | 0,0107 | 0,815 | 0,0089 |
| 4 | 1,28 | 0,0089 | 0,845 | 0,0074 |
| 5 | 1,03 | 0,0082 | 0,853 | 0,0070 |
| 6 | 0,87 | 0,0077 | 0,861 | 0,0062 |
| 7 | 0,77 | 0,0075 | 0,866 | 0,0061 |
| 8 | 0,72 | **0,0073** | 0,870 | 0,0060 |

La mejora se aplana en las últimas épocas, cuando la tasa de aprendizaje se acerca a su valor final. La parada temprana no se activó y el mejor modelo es el de la época 8.

## 5. Resultados en validación

| Conjunto | CER | Exactitud de secuencia | CER de forma |
|---|---|---|---|
| Predictor trivial (validación) | ≈ 0,91 | – | – |
| Entrenamiento (2 000 aleatorias) | 0,0057 | 0,8995 | 0,0044 |
| Validación, submuestra (2 000) | 0,0073 | 0,870 | 0,0060 |
| **Validación completa (8 772)** | **0,0067** | **0,8805** | **0,0054** |

Se evaluaron 240 867 tokens de referencia. El CER recalculado sobre la submuestra (0,0073) coincide con el registrado durante el entrenamiento. La submuestra resultó algo más difícil que la validación completa.

### Sesgo, varianza y ajuste

- **Varianza baja:** el CER pasa de 0,0057 (entrenamiento, en modo de inferencia y sin *dropout*) a 0,0067 (validación), una diferencia relativa del 15 %, y la exactitud de secuencia de 0,8995 a 0,8805. El modelo no memoriza el entrenamiento.
- **Sesgo bajo, pero no nulo:** el modelo sigue errando en imágenes de entrenamiento (el 10 % de esas secuencias no coincide exactamente). Parte de ese error proviene de inconsistencias de las etiquetas (sección 7) y parte de símbolos pequeños que el modelo aún no resuelve (sección 6).
- **Conclusión:** ajuste adecuado, sin sobreajuste. La regularización o el aumento de datos no abordarían el error dominante en esta distribución de datos, lo que orienta la primera mejora (EXP-01) hacia la resolución del modelo.

## 6. Análisis de errores (validación completa)

Hay 1 604 errores en 1 048 de las 8 772 secuencias.

### Por tipo

| Tipo | Conteo | Tasa sobre tokens | Parte de los errores |
|---|---|---|---|
| Omitidos (borrado) | 700 | 0,0029 | 43,6 % |
| Insertados | 493 | 0,0020 | 30,7 % |
| Mismo símbolo, altura incorrecta | 266 | 0,0011 | 16,6 % |
| Símbolo incorrecto | 145 | 0,0006 | 9,0 % |
| **Total** | **1 604** | **0,0067** | 100 % |

El 74 % de los errores son omisiones o inserciones, y solo el 17 % son errores de altura.

### Errores de altura

Son errores pequeños: el 62,0 % (165 casos) se desplaza un paso del pentagrama, el 34,2 % (91) dos pasos y el 3,8 % (10) entre tres y seis. El 81 % (215 de 266) corresponde a notas de adorno (`gracenote`). Las confusiones más frecuentes son `gracenote.sixteenth-L4` → `S3` (15 casos) y `L4` → `S4` (10 casos).

### Recall por tipo de símbolo

| Tipo | Soporte | Recall | Recall de forma | Omitidos |
|---|---|---|---|---|
| Nota | 133 595 | 0,9991 | 0,9994 | 6 |
| Barra de compás | 32 594 | 0,9963 | 0,9963 | 116 |
| Alteración | 17 331 | 0,9710 | 0,9717 | 471 |
| Silencio | 13 114 | 0,9998 | 0,9998 | 1 |
| Puntillo | 11 826 | 0,9994 | 0,9997 | 2 |
| Cifra | 11 175 | 0,9992 | 0,9992 | 1 |
| Clave | 8 849 | 0,9994 | 0,9994 | 3 |
| Indicación de compás | 4 796 | 1,0000 | 1,0000 | 0 |
| Ligadura | 2 594 | 0,9680 | 0,9680 | 77 |
| Silencio multicompás | 2 294 | 0,9996 | 0,9996 | 0 |
| Nota de adorno | 2 257 | 0,8848 | 0,9801 | 22 |
| Calderón | 442 | 0,9977 | 0,9977 | 1 |

### Familias con menor recall (al menos 100 apariciones)

| Familia | Soporte | Recall | Recall de forma | Omitidos |
|---|---|---|---|---|
| `gracenote.sixteenth` | 700 | 0,7471 | 0,9671 | 17 |
| `accidental.natural` | 1 489 | 0,7683 | 0,7696 | 336 |
| `slur.end` | 1 297 | 0,9645 | 0,9645 | 42 |
| `slur.start` | 1 297 | 0,9715 | 0,9715 | 35 |
| `accidental.flat` | 7 385 | 0,9839 | 0,9842 | 110 |

Las notas de duración común (negras, corcheas, blancas y notas con haz) superan 0,999.

### Omisiones e inserciones

- **Omisiones:** las alteraciones suman 471 (29 % de todos los errores): becuadros 336, bemoles 110 y sostenidos 25. Además se omiten 116 barras de compás y 77 ligaduras (42 finales y 35 de inicio).
- **Inserciones:** la más frecuente es `barline-L1` (216), seguida de ligaduras (`slur.start` y `slur.end`) en distintas posiciones.

### Por longitud de secuencia y ancho de imagen

| Grupo | Muestras | CER | Exactitud de secuencia |
|---|---|---|---|
| 1 a 20 tokens | 1 296 | 0,0075 | 0,9128 |
| 21 a 40 tokens | 7 145 | 0,0063 | 0,8810 |
| 41 a 60 tokens | 329 | 0,0105 | 0,7447 |
| 61 a 75 tokens | 2 | 0,0163 | 0,5000 |
| Ancho 361 a 918 px | 1 764 | 0,0071 | 0,8883 |
| Ancho 919 a 1 072 px | 1 752 | 0,0063 | 0,8864 |
| Ancho 1 073 a 1 219 px | 1 750 | 0,0057 | 0,9006 |
| Ancho 1 220 a 1 433 px | 1 753 | 0,0064 | 0,8859 |
| Ancho 1 434 a 3 417 px | 1 753 | 0,0078 | 0,8414 |

El efecto del ancho es leve, salvo en el quintil de imágenes más anchas. El grupo de 61 a 75 tokens solo tiene 2 muestras y no es concluyente.

## 7. Inconsistencias en las etiquetas

La inspección de ejemplos mostró dos peculiaridades de las referencias que no son errores del modelo:

- **Barra final.** En 219 secuencias con error, la predicción termina con una `barline-L1` que la referencia no incluye, y en 111 ocurre lo contrario. Solo el 52,3 % de las referencias de validación termina con barra. En los dos ejemplos revisados visualmente (índices 7666 y 702) la barra **sí está dibujada** en la imagen: el modelo acierta y la referencia la omite. Estas 330 secuencias equivalen a cerca de un tercio de las secuencias con error y, como máximo, al 20 % de los errores a nivel de token. Sin ese efecto el CER sería del orden de 0,0053 (cota aproximada: (1 604 − 330) / 240 867); el valor exacto está pendiente.
- **Haces de longitud cero.** Algunas notas se etiquetan `note.beamedRight0`, `note.beamedBoth0` o `note.beamedLeft0` (un haz sin barras), pero en las imágenes revisadas (índices 785 y 643) son negras sueltas. El modelo predice `note.quarter`, que es lo visible. El conteo de estos tokens en validación está pendiente.

Otros ejemplos revisados son errores del modelo: una ligadura evidente que se omite (índice 7509) y una ligadura casi superpuesta con la quinta línea del pentagrama, visualmente ambigua (índice 572).

## 8. Interpretación

1. El modelo resuelve la tarea con poco error: CER de 0,0067 frente a ≈ 0,91 de la línea base, con el 88 % de los pentagramas sin ningún error.
2. Los errores restantes no son de altura sino de presencia o ausencia de símbolos pequeños o pegados a otros (becuadros, bemoles, ligaduras, notas de adorno) y de la barra final.
3. Una parte del error es de las etiquetas y establece un piso para el CER con estas referencias. Por eso se prevé reportar también el CER sin barra final.
4. Hipótesis para EXP-01: reducir la compresión horizontal de ×8 a ×4, de modo que cada fotograma de la CTC cubra menos imagen y menos símbolos compitan por el mismo fotograma. Es una hipótesis; el diseño y el criterio de éxito están en [`architecture.md`](architecture.md), sección 7.

## 9. Reproducibilidad de la evaluación

- El análisis se ejecuta en `notebooks/04_evaluation.ipynb`, cuyas salidas (tablas y ejemplos con imagen) quedan guardadas en el propio notebook.
- Los archivos de detalle (`metrics_val.json`, `recall_by_family.csv`, `recall_by_kind.csv` y `errors_val.csv`) se generan en `artifacts/eval_exp00_val/` de Drive y no se versionan en Git. La evaluación del conjunto de validación completo tardó unos 28 min en CPU de Colab.
- Para evaluar otro experimento basta cambiar `EXP_ID` y `RUN_DIR`, y usar la CRNN con el factor de reducción del experimento.

## 10. Pendiente

- Valor exacto del CER sin barra final y conteo de etiquetas `beamed…0` (celda 10.3 del notebook 04).
- Evaluación de EXP-01 sobre validación completa y comparación con EXP-00 con las mismas particiones y métricas.
- Selección del modelo final y única evaluación sobre el conjunto de prueba.
