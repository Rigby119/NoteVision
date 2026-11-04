# Limitaciones y trabajo futuro

Este documento reúne lo que los resultados de NoteVision **no** permiten afirmar y los límites del alcance actual. Las cifras que se mencionan están en [`evaluation.md`](evaluation.md).

---

## 1. Datos

- **Licencia:** licencia explícita del dataset no encontrada; uso académico con atribución. Por eso los datos no se redistribuyen en el repositorio.
- **Cobertura de etiquetas.** Se usa la representación agnóstica de PrIMuS (758 tokens, longitud máxima de 75). Los símbolos que no estén en ese vocabulario no pueden predecirse.
- **Clases raras:** 90 de los 758 tokens aparecen menos de 5 veces. Algunos quedarán sin ejemplos en validación o prueba, y el modelo difícilmente los aprenderá.
- **Datos sintéticos:** las imágenes están renderizadas con Verovio, son monódicas y de un solo pentagrama. No hay garantía de que el modelo funcione en escaneos, fotografías o partituras manuscritas.
- **Inconsistencias en las etiquetas.**
  - *Barra final:* solo el 52,3 % de las referencias de validación termina con `barline-L1`, y en los ejemplos revisados la barra está dibujada aunque la referencia la omita. Afecta a unas 330 secuencias de validación (219 con barra de más en la predicción y 111 con barra de menos).
  - *Haces de longitud cero:* algunas notas se etiquetan `beamed…0` aunque en la imagen son negras sueltas.
  - Estas inconsistencias imponen un piso al CER con estas referencias; una parte del error medido no es atribuible al modelo.

## 2. Modelo y resultados

- **Símbolos pequeños.** El modelo omite o inventa con más frecuencia becuadros (recall 0,768), bemoles, ligaduras (0,965 a 0,972) y notas de adorno, sobre todo las semicorcheas (0,747 con posición; 0,967 si se ignora la posición). Los errores de altura que quedan son casi siempre de uno o dos pasos.
- **Una sola corrida por experimento.** No se mide la variación entre semillas; en las pruebas cortas de depuración se observó variación entre ejecuciones. Por eso los criterios de éxito de los experimentos piden una mejora relativa mínima (10 % en EXP-01).
- **Selección con parte de la validación.** El mejor modelo se elige con una submuestra de 2 000 imágenes incluida en la validación completa, por lo que la cifra de validación completa no es totalmente independiente de la selección.
- **Conjunto de prueba sin usar.** Hasta ahora no hay métricas de prueba. Se evaluará una sola vez, con el modelo final.
- **Decodificación *greedy*.** No se usa búsqueda por haz ni modelo de lenguaje musical, por lo que no se aprovechan reglas de la notación (por ejemplo, que un compás complete su duración).
- **Restricción de CTC.** Cada fotograma emite un solo símbolo, y dos tokens idénticos consecutivos necesitan un blanco entre ellos. Con la reducción horizontal ×8 esto puede explicar parte de las omisiones de símbolos pegados a otros, aunque es una hipótesis que EXP-01 pondrá a prueba.

## 3. Alcance

- **Un pentagrama monofónico impreso.** Quedan fuera las partituras polifónicas, los sistemas con varios pentagramas, los PDF multipágina y la notación manuscrita.
- **Posiciones relativas, no música completa.** El modelo predice símbolo y posición en el pentagrama. La altura real depende de la clave, de la armadura y de las alteraciones; `predict.py --pitch` aplica solo la clave y **no** aplica armadura ni alteraciones.
- **Clave asumida.** Si no se predice ninguna clave, `predict.py` asume la clave de sol y lo marca explícitamente como **supuesto**, nunca como detección confirmada.
- **Ancho máximo.** Se entrenó con imágenes cuyo ancho, tras reescalar a 128 px de alto, es menor que 3 600 px. `predict.py` advierte cuando una imagen lo supera.

## 4. Entorno y reproducibilidad

- El entrenamiento se hizo en Google Colab (GPU T4), con TensorFlow 2.21.0 y Keras 3.13.2. Las sesiones pueden interrumpirse y la velocidad varía entre sesiones, por lo que el entrenamiento guarda puntos de control y puede reanudarse.
- Las GPU no son totalmente deterministas: repetir un entrenamiento con la misma semilla puede dar resultados ligeramente distintos.
- Los pesos cargados por `predict.py` requieren versiones compatibles de TensorFlow y Keras con las usadas al guardarlos.
- Los datos y los artefactos grandes (`artifacts/`, salvo `artifacts/model/`) viven en Drive y no se versionan en Git.

## 6. Trabajo futuro

- **EXP-01:** reducción horizontal de ×8 a ×4 y comparación con EXP-00.
- Métrica sin barra final y limpieza o normalización de las etiquetas inconsistentes.
- Aumento de datos y evaluación con Camera-PrIMuS, para medir robustez ante distorsiones.
- Decodificación con búsqueda por haz o restricciones de notación.
- Lectura de PDF multipágina (conversión de páginas a imágenes, localización de pentagramas y compases, y unión de las transcripciones).
- Partituras manuscritas.
