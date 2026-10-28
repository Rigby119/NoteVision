# NoteVision

Reconocimiento de símbolos musicales en imágenes de partituras impresas mediante aprendizaje profundo.

> **Estado: planificación e implementación inicial (actualizado el 26 de octubre de 2026).**
> Este repositorio aún no contiene un modelo entrenado, resultados experimentales ni comandos verificados. Las secciones marcadas como *pendiente* se completarán a medida que avance el proyecto.

---

## 1. Descripción

NoteVision es un proyecto académico individual que busca desarrollar un modelo de *deep learning* capaz de leer una imagen de un compás musical impreso y producir una **secuencia estructurada de eventos musicales**.

## 2. Objetivo general

Entrenar y evaluar un modelo de aprendizaje profundo, implementado con un framework, que reconozca símbolos musicales en imágenes de partituras impresas con un dataset real; mejorar su desempeño mediante experimentos controlados y documentados; y ofrecer predicciones reproducibles desde la consola.

## 3. Problema que aborda

Las partituras impresas existen como imágenes, pero no como información musical estructurada que un programa pueda manipular. El reconocimiento óptico de música (OMR, *Optical Music Recognition*) busca decodificar esa notación a partir de imágenes. Este proyecto aborda una versión acotada del problema: una imagen recortada, impresa y de un solo pentagrama o compás, y una salida secuencial.

## 4. Alcance de la primera versión

**Fijo**
- Entrada: imagen PNG/JPG recortada de un compás musical impreso.
- Salida: secuencia estructurada de símbolos musicales.
- Reconocimiento de **alturas relativas al pentagrama** (primera capacidad que se valida).
- Predicción mediante **interfaz de consola**.

**Condicional (sujeto a las etiquetas del dataset)**
- Duraciones y silencios, solo si hay etiquetas suficientes para entrenar y evaluar.
- Claves de sol y de fa: se separará la posición relativa de la interpretación musical absoluta. Si no se detecta ninguna clave y se asume la de sol, esto se marcará explícitamente como **supuesto**, nunca como detección confirmada.

**Fuera de la primera versión**
- Interfaz web (opcional, solo tras cumplir el alcance fijo).
- Lectura de PDF multipágina (trabajo futuro: requeriría convertir páginas a imágenes, localizar pentagramas y compases, y unir las transcripciones).

## 5. Dataset

**Candidato principal: PrIMuS** (*Printed Images of Music Staves*).

Según su documentación pública, contiene 87 678 incipits musicales reales, monódicos y de un solo pentagrama, extraídos de RISM, con las partituras renderizadas como imágenes y representaciones simbólicas codificadas como secuencias. También existe una extensión, **Camera-PrIMuS**, cuyas imágenes se distorsionan para simular fotografías.

## 6. Arquitectura tentativa

Se considera una arquitectura **convolutiva con un componente secuencial**, por ejemplo **CNN + BiLSTM + CTC**:

- **CNN:** extracción de características visuales de la imagen.
- **Componente secuencial (p. ej., BiLSTM):** modelado del orden de los símbolos de izquierda a derecha.
- **CTC (*Connectionist Temporal Classification*):** función de pérdida que permite entrenar con secuencias de etiquetas sin alineamiento símbolo a símbolo.

El framework de aprendizaje profundo aún **no ha sido seleccionado**.

## 7. Plan de entrenamiento, validación, prueba y mejora

| Etapa | Qué se hará | Qué se registrará |
|---|---|---|
| Línea base (EXP-00) | Entrenar el primer modelo; depurar primero con una muestra reproducible y luego ampliar | Métricas, configuración, tiempo de entrenamiento y ejemplos de errores |
| Experimento 1 (EXP-01) | Probar el efecto de la regularización o del aumento de datos, manteniendo lo demás constante | Qué cambió y si mejoró la generalización |
| Experimento 2 (EXP-02), si hay tiempo | Ajuste de hiperparámetros | Diferencias frente a la línea base |
| Experimento 3 (EXP-03), condicional | Otra arquitectura, solo si los resultados anteriores lo justifican | Si el beneficio compensa la complejidad |
| Comparación final | Modelo inicial frente al mejor candidato, con las mismas particiones y métricas | Tabla comparativa y conclusiones |
| Predicción | Probar con imágenes no vistas | Imagen, predicción, referencia esperada y errores observados |

Los experimentos se registrarán en `experiments/results.csv` y en `docs/`.

## 8. Métricas previstas

Las métricas definitivas dependerán de las etiquetas disponibles en el dataset. Punto de partida:

- **Tasa de error de edición de secuencia (CER):** cuánto difiere la secuencia predicha de la referencia.
- **Exactitud de secuencia:** proporción de ejemplos cuya secuencia completa coincide con la referencia.
- **Precisión, recall y F1 por clase:** solo si pueden evaluarse por separado alturas, duraciones o silencios.
- **Pérdida de entrenamiento y de validación:** para detectar sobreajuste y problemas de convergencia.

## 9. Estructura inicial del repositorio

```
NoteVision/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── code/
│   ├── dataset_audit.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
├── experiments/
│   └── results.csv
└── docs/
    ├── dataset.md
    ├── architecture.md
    ├── evaluation.md
    └── limitations.md
```

Esta es la estructura propuesta; los archivos se irán creando durante el proyecto. Los pesos entrenados pueden ser demasiado grandes para GitHub: en ese caso se documentará cómo obtenerlos y dónde colocarlos localmente.

## 10. Requisitos de software previstos

Aún **no se han seleccionado versiones**. Se registrarán en `requirements.txt` cuando se configure el entorno.

- Python
- Un framework de aprendizaje profundo (por definir)
- Bibliotecas de cálculo numérico y procesamiento de imágenes
- Jupyter para los notebooks

## 11. Limitaciones y trabajo futuro

**Limitaciones previstas**
- La cobertura de etiquetas del dataset puede limitar qué se puede aprender y evaluar (duraciones, silencios, claves).
- El dataset candidato contiene partituras impresas y sintéticas; el desempeño en imágenes reales o distintas no está garantizado.

**Trabajo futuro (fuera de la primera versión)**
- Interfaz web.
- Lectura de PDF multipágina.
- Partituras manuscritas.

## 12. Referencias

- Calvo-Zaragoza, J., & Rizo, D. (2018a). Camera-PrIMuS: Neural end-to-end optical music recognition on realistic monophonic scores. En *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR 2018)* (pp. 248–255). https://ismir2018.ircam.fr/doc/pdfs/33_Paper.pdf
- Calvo-Zaragoza, J., & Rizo, D. (2018b). End-to-end neural optical music recognition of monophonic scores. *Applied Sciences, 8*(4), Article 606. https://doi.org/10.3390/app8040606
- Calvo-Zaragoza, J.; Rizo, D. (2018). *Camera-PrIMuS: Neural End-to-End Optical Music Recognition on Realistic Monophonic Scores*. Proceedings of the 19th ISMIR Conference, París, pp. 248–255.
- Repositorio de código asociado a PrIMuS: https://github.com/OMR-Research/tf-end-to-end
- Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. En *Proceedings of the 23rd International Conference on Machine Learning* (pp. 369–376). Association for Computing Machinery. https://doi.org/10.1145/1143844.1143891
- Optical Music Recognition Research. (s. f.). *tf-end-to-end* [Repositorio de código]. GitHub. https://github.com/OMR-Research/tf-end-to-end
- Pattern Recognition and Artificial Intelligence Group. (s. f.). *The Printed Images of Music Staves (PrIMuS) dataset*. Universidad de Alicante. https://grfia.dlsi.ua.es/primus/
- Shi, B., Bai, X., & Yao, C. (2017). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 39*(11), 2298–2304. https://doi.org/10.1109/TPAMI.2016.2646371
