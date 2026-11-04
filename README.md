# NoteVision

Reconocimiento de símbolos musicales en imágenes de partituras impresas mediante aprendizaje profundo.

> **Estado: modelo inicial (EXP-00) entrenado y evaluado en validación; EXP-01 en curso (actualizado el 3 de noviembre de 2026).**

---

## 1. Descripción

NoteVision es un proyecto académico individual que busca desarrollar un modelo de *deep learning* capaz de leer una imagen de un compás musical impreso y producir una **secuencia estructurada de eventos musicales**.

## 2. Objetivo general

Entrenar y evaluar un modelo de aprendizaje profundo, implementado con TensorFlow/Keras, que reconozca símbolos musicales en imágenes de partituras impresas con un dataset real; mejorar su desempeño mediante experimentos controlados y documentados; y ofrecer predicciones reproducibles desde la consola.

## 3. Problema que aborda

Las partituras impresas existen como imágenes, pero no como información musical estructurada que un programa pueda manipular. El reconocimiento óptico de música (OMR, *Optical Music Recognition*) busca decodificar esa notación a partir de imágenes. Este proyecto aborda una versión acotada del problema: una imagen recortada, impresa y de un solo pentagrama o compás, y una salida secuencial.

## 4. Alcance de la primera versión

**Fijo**
- Entrada: imagen PNG/JPG recortada de un compás musical impreso.
- Salida: secuencia estructurada de símbolos musicales.
- Reconocimiento de **alturas relativas al pentagrama** (primera capacidad que se valida).
- Predicción mediante **interfaz de consola**.
- Duraciones y silencios.
- Claves de sol y de fa: se separará la posición relativa de la interpretación musical absoluta. Si no se detecta ninguna clave y se asume la de sol, esto se marcará explícitamente como **supuesto**, nunca como detección confirmada.

**Fuera de la primera versión**
- Interfaz web (opcional, solo tras cumplir el alcance fijo).
- Lectura de PDF multipágina (trabajo futuro: requeriría convertir páginas a imágenes, localizar pentagramas y compases, y unir las transcripciones).

## 5. Dataset

**Dataset seleccionado: PrIMuS** (*Printed Images of Music Staves*).

Según su documentación pública, contiene 87 678 incipits musicales reales, monódicos y de un solo pentagrama, extraídos de RISM, con las partituras renderizadas como imágenes y representaciones simbólicas codificadas como secuencias. También existe una extensión, **Camera-PrIMuS**, cuyas imágenes se distorsionan para simular fotografías.

Se usa la representación **agnóstica** (símbolo y posición en el pentagrama, por ejemplo `note.quarter-S2`), con un vocabulario de 758 tokens y etiquetas de hasta 75 tokens. Las particiones son 80/10/10 por grupo, para evitar fuga de datos (semilla 42): 70 138 imágenes de entrenamiento, 8 772 de validación y 8 768 de prueba.

**Licencia y uso:** licencia explícita del dataset no encontrada; uso académico con atribución. Los datos no se redistribuyen en este repositorio: se documenta cómo descargarlos desde la fuente original (ver [`dataset.md`](docs/dataset.md)).

**Calidad de las etiquetas:** la evaluación detectó inconsistencias en las referencias (por ejemplo, en la barra de compás final); se describen en [`evaluation.md`](docs/evaluation.md) y [`limitations.md`](docs/limitations.md).

## 6. Arquitectura

Se utiliza una arquitectura **convolutiva con un componente secuencial**, **CNN + BiLSTM + CTC** (CRNN):

- **CNN:** extracción de características visuales de la imagen.
- **Componente secuencial (BiLSTM):** modelado del orden de los símbolos de izquierda a derecha.
- **CTC (*Connectionist Temporal Classification*):** función de pérdida que permite entrenar con secuencias de etiquetas sin alineamiento símbolo a símbolo.

El framework utilizado es **TensorFlow/Keras**. La arquitectura implementada (CRNN) tiene cuatro bloques convolucionales con reducción horizontal ×8, dos capas BiLSTM de 128 unidades por dirección y una capa densa de 759 clases (758 tokens agnósticos más el blanco CTC, en el índice 0): **1 896 407 parámetros entrenables** (≈ 8 MB). La decodificación es *greedy* CTC. El detalle de capas, entrenamiento y la propuesta de EXP-01 está en [`architecture.md`](docs/architecture.md).

## 7. Plan de entrenamiento, validación, prueba y mejora

| Etapa | Qué se hará | Qué se registrará | Estado |
|---|---|---|---|
| Línea base (EXP-00) | Entrenar el primer modelo; depurar primero con una muestra reproducible y luego ampliar | Métricas, configuración, tiempo de entrenamiento y ejemplos de errores | Completado |
| Experimento 1 (EXP-01) | Probar si reducir la compresión horizontal de ×8 a ×4 mejora el reconocimiento de símbolos pequeños (hipótesis derivada del análisis de errores), manteniendo lo demás constante | Qué cambió y si mejoró el CER y el recall por familia | En curso |
| Experimento 2 (EXP-02), si hay tiempo | Ajuste de hiperparámetros | Diferencias frente a la línea base | Pendiente |
| Experimento 3 (EXP-03), condicional | Otra arquitectura, solo si los resultados anteriores lo justifican | Si el beneficio compensa la complejidad | Pendiente |
| Comparación final | Modelo inicial frente al mejor candidato, con las mismas particiones y métricas; el conjunto de prueba se evalúa una sola vez | Tabla comparativa y conclusiones | Pendiente |
| Predicción | Probar con imágenes no vistas | Imagen, predicción, referencia esperada y errores observados | Pendiente |

El protocolo es común a todos los experimentos: 8 épocas, el mismo calendario de tasa de aprendizaje, una submuestra fija de 2 000 imágenes de validación y el mismo criterio de mejor modelo. La regularización o el aumento de datos no fueron la primera mejora porque EXP-00 no muestra sobreajuste (ver [`evaluation.md`](docs/evaluation.md)).

Los experimentos se registrarán en `experiments/results.csv` y en `docs/`.

## 8. Métricas

Con las etiquetas agnósticas de PrIMuS (símbolo y posición en el pentagrama, incluyendo duraciones, silencios y claves), las métricas son:

- **CER (principal):** distancia de edición entre la secuencia predicha y la de referencia, sumada sobre todo el conjunto, dividida entre el total de tokens de referencia.
- **Exactitud de secuencia:** proporción de ejemplos cuya secuencia completa coincide con la referencia.
- **CER de forma (`cer_shape`):** el mismo cálculo comparando solo la familia del símbolo (nota, silencio, clave, etc.) e ignorando la posición vertical; separa los errores de símbolo de los de altura.
- **Recall por familia de símbolo:** aciertos entre apariciones, a partir del alineamiento.
- **CER sin barra final (secundaria):** el CER ignorando una barra de compás final en referencia y predicción, para neutralizar una inconsistencia de las etiquetas.
- **Pérdida de entrenamiento y de validación (CTC):** para detectar sobreajuste y problemas de convergencia.

**Línea base de comparación:** predictores triviales sin imagen (la secuencia más frecuente y la moda por posición), con CER ≈ 0,91 en validación y prueba.

### 8.1. Resultados de EXP-00 (validación)

| Conjunto | CER | Exactitud de secuencia | CER de forma |
|---|---|---|---|
| Predictor trivial (validación) | ≈ 0,91 | – | – |
| Entrenamiento (2 000 aleatorias) | 0,0057 | 0,8995 | 0,0044 |
| Validación, submuestra (2 000) | 0,0073 | 0,870 | 0,0060 |
| **Validación completa (8 772)** | **0,0067** | **0,8805** | **0,0054** |

- Entrenamiento de 8 épocas en Google Colab (GPU T4): 3,94 h; mejor modelo en la época 8.
- La brecha entre entrenamiento y validación es pequeña (no hay sobreajuste).
- El 74 % de los errores son omisiones o inserciones, sobre todo de becuadros, bemoles, ligaduras, notas de adorno y la barra final; solo el 17 % son errores de altura, casi siempre de uno o dos pasos.
- El conjunto de prueba aún no se ha utilizado: se evaluará una sola vez, al elegir el modelo final.

Detalle completo en [`evaluation.md`](docs/evaluation.md).

## 9. Estructura del repositorio

```
NoteVision/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── artifacts/
│   └── model/
│       └── best.weights.h5
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── code/
│   └── predict.py
├── experiments/
│   └── results.csv
└── docs/
    ├── dataset.md
    ├── architecture.md
    ├── evaluation.md
    ├── limitations.md
    └── NoteVision.pdf          (paper académico)
```

El desarrollo se realiza primero en los notebooks (Google Colab); `code/` contiene los scripts que se ejecutan directamente desde la consola.

De `artifacts/` solo se versiona `artifacts/model/` (pesos y vocabulario, ≈ 8 MB); el resto de los artefactos (particiones, puntos de control, resultados de evaluación) vive en Google Drive y no se incluye en Git.

## 10. Requisitos de software

El entrenamiento se realiza en **Google Colab**; la predicción por consola se probará también en local (Windows). Las versiones usadas en Colab son TensorFlow 2.21.0 y Keras 3.13.2; las versiones exactas de Python y de las bibliotecas se registrarán en `requirements.txt` a partir del entorno que efectivamente se use cuando esté estable.

- Python 3
- TensorFlow/Keras
- Bibliotecas de cálculo numérico y procesamiento de imágenes (NumPy, pandas, Pillow)
- Jupyter / Google Colab para los notebooks

## 11. Uso

### 11.1. Predicción por consola

`code/predict.py` es autocontenido: define la CRNN, aplica el mismo preprocesamiento del entrenamiento, carga los pesos y el vocabulario de `artifacts/model/` y entrega la secuencia de tokens de una o varias imágenes PNG/JPG.

```
python code/predict.py ruta/imagen.png
python code/predict.py imagen1.png imagen2.jpg --json
python code/predict.py imagen.png --pitch
```

- `--json`: imprime la salida en JSON.
- `--pitch`: añade letra y octava a las notas a partir de la posición y la clave; **no** aplica armadura ni alteraciones.
- Si no se predice ninguna clave, el programa **asume la clave de sol y lo marca explícitamente como supuesto**, nunca como detección.
- Si la imagen, tras reescalarla, es más ancha que cualquiera de las usadas en el entrenamiento, el programa lo advierte.

### 11.2. NoteVision-Web (aplicación web)

[**NoteVision-Web**](https://github.com/Rigby119/NoteVision-Web) es un proyecto complementario, en un repositorio aparte, que permite visualizar el resultado del modelo: se abre una partitura (PDF o imagen), se recorta un sistema, se ejecuta la CRNN y la predicción se muestra como notación musical y se reproduce con audio sintetizado (con exportación a WAV y MIDI). Está hecha con React y FastAPI, procesa todo en memoria y usa los mismos pesos y vocabulario (`best.weights.h5` y `vocab_agnostic.csv`) y las mismas versiones (TensorFlow 2.21.0 y Keras 3.13.2). La instalación y el uso se describen en el README de ese repositorio.

## 12. Limitaciones y trabajo futuro

**Limitaciones** (detalle en [`limitations.md`](docs/limitations.md))
- Las imágenes de PrIMuS son impresas y sintéticas; el desempeño en fotografías o escaneos no está garantizado.
- Las etiquetas del dataset tienen inconsistencias (barra final, haces de longitud cero) que limitan el CER alcanzable.
- El modelo aún omite o inventa símbolos pequeños (becuadros, ligaduras, notas de adorno).
- Cada experimento es una sola corrida, y el conjunto de prueba no se ha utilizado.
- Alcance: un pentagrama monofónico impreso; las posiciones son relativas al pentagrama.

**Trabajo futuro (fuera de la primera versión)**
- Evaluación con Camera-PrIMuS.
- Lectura automática de PDF multipágina: NoteVision-Web ya abre PDF y permite recortar un sistema a mano; falta localizar pentagramas y compases de forma automática y unir las transcripciones.
- Partituras manuscritas.

## 13. Referencias

- Calvo-Zaragoza, J., & Rizo, D. (2018a). Camera-PrIMuS: Neural end-to-end optical music recognition on realistic monophonic scores. En *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR 2018)* (pp. 248–255). https://ismir2018.ircam.fr/doc/pdfs/33_Paper.pdf
- Calvo-Zaragoza, J., & Rizo, D. (2018b). End-to-end neural optical music recognition of monophonic scores. *Applied Sciences, 8*(4), Article 606. https://doi.org/10.3390/app8040606
- Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. En *Proceedings of the 23rd International Conference on Machine Learning* (pp. 369–376). Association for Computing Machinery. https://doi.org/10.1145/1143844.1143891
- Optical Music Recognition Research. (s. f.). *tf-end-to-end* [Repositorio de código]. GitHub. https://github.com/OMR-Research/tf-end-to-end
- Pattern Recognition and Artificial Intelligence Group. (s. f.). *The Printed Images of Music Staves (PrIMuS) dataset*. Universidad de Alicante. https://grfia.dlsi.ua.es/primus/
- Shi, B., Bai, X., & Yao, C. (2017). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 39*(11), 2298–2304. https://doi.org/10.1109/TPAMI.2016.2646371
