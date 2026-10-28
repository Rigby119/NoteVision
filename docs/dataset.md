# Dataset: PrIMuS

## 1. Fuente

PrIMuS (*Printed Images of Music Staves*) es un dataset del grupo PRAIG de la Universidad de Alicante: https://grfia.dlsi.ua.es/primus/

- Los incipits musicales provienen de un export de la base de datos RISM.
- Las imágenes y las representaciones simbólicas se generaron automáticamente con Verovio.
- La documentación del dataset menciona tres tipografías musicales (Leipzig, Bravura y Gootville).

## 2. Licencia y condiciones de uso

**Licencia explícita del dataset no encontrada; uso académico con atribución.**

Lo que sí está confirmado:
- El dataset se descarga de forma pública y gratuita desde el sitio del grupo.
- Se debe citar: Calvo-Zaragoza, J., & Rizo, D. (2018). *End-to-end neural optical music recognition of monophonic scores*. Applied Sciences, 8(4), 606. https://doi.org/10.3390/app8040606

Lo que NO se asume:
- La licencia CC BY 4.0 corresponde al artículo científico, y la licencia MIT al código de `tf-end-to-end`. Ninguna de las dos se toma como licencia de los archivos del dataset.
- No se afirma que los datos se puedan modificar o redistribuir libremente.

Decisión del proyecto: los datos **no se suben al repositorio** (`data_raw/` y `primus/` están en `.gitignore`). Solo se documenta cómo obtenerlos.

## 3. Contenido de cada muestra

Cada muestra es una carpeta `{id}` con:

| Archivo | Contenido |
|---|---|
| `{id}.png` | Imagen del incipit |
| `{id}.agnostic` | Etiquetas agnósticas (las que usa el proyecto) |
| `{id}.semantic` | Etiquetas semánticas |
| `{id}.mei` | Codificación MEI |
| `{id}.mid` | Archivo MIDI |
| `regular_pae.pae` | Codificación Plaine & Easie original |

Los archivos `._*` que aparecen dentro del `.tgz` son metadatos de macOS y se ignoran. La auditoría detectó que contarlos por extensión infla los conteos de `.png` y `.mei`.

## 4. Cifras de la auditoría

| Dato | Valor |
|---|---|
| Muestras totales | 87 678 |
| `package_aa` | 43 594 |
| `package_ab` | 44 084 |
| IDs repetidos entre paquetes | 0 |
| Muestras sin imagen o sin etiqueta agnóstica | 0 |

Resumen guardado en `artifacts/audit_summary.json`.

## 5. Imágenes

- Modo de color: `P` (paleta), por lo que se convierten a escala de grises.
- Tamaño (muestra aleatoria de 2000): ancho 308–2003 px (mediana 1272) y alto 74–209 px (mediana 143).
- Son tiras largas, con una proporción ancho/alto de aproximadamente 9:1.

## 6. Etiquetas agnósticas

- Formato de cada token: `símbolo.duración-posición`, por ejemplo `note.quarter-L4`, `rest.eighth-L3`, `clef.G-L2`.
- La posición es una línea (`L`) o un espacio (`S`) del pentagrama. Existen valores negativos (`S-2`, `L-1`) para posiciones por debajo del pentagrama.
- Longitud de las secuencias: de 3 a 75 tokens (media 27.5).
- Vocabulario: 758 tokens distintos y 73 familias de símbolo.
- Las notas con barra (`note.beamedLeft1`, `note.beamedBoth2`, etc.) codifican el número de barras, no solo la duración.
- Duraciones, silencios y claves (`clef.G`, `clef.C`) están presentes en las etiquetas, por lo que forman parte del alcance fijo.

Vocabulario completo en `artifacts/vocab_agnostic.csv`.

## 6.1 Representación de salida

El token agnóstico completo es la unidad de salida: 758 clases más el blanco de CTC (759 salidas).

## 7. Riesgo de fuga de datos

- 3157 secuencias de etiquetas repetidas (7118 muestras, alrededor de 8 %).
- 61 772 obras distintas (prefijo del ID antes del guion); algunas tienen hasta 59 incipits.
- Se formaron grupos conectados por obra y por secuencia idéntica: 58 577 grupos, el mayor con 194 muestras (0,22 % del dataset).
- Las particiones de entrenamiento, validación y prueba se harán **por grupo**, con semilla fija (42).

Índice con grupos en `artifacts/index_agnostic.csv`.