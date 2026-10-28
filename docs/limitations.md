# Limitaciones

## Datos

- **Licencia:** licencia explícita del dataset no encontrada; uso académico con atribución. Por eso los datos no se redistribuyen en el repositorio.
- **Clases raras:** 90 de los 758 tokens aparecen menos de 5 veces. Algunos quedarán sin ejemplos en validación o prueba, y el modelo difícilmente los aprenderá.
- **Datos sintéticos:** las imágenes están renderizadas con Verovio, son monódicas y de un solo pentagrama. No hay garantía de que el modelo funcione en escaneos, fotografías o partituras manuscritas.
- **Camera-PrIMuS:** queda fuera de la primera versión.

## Preprocesamiento y modelo

- Las imágenes están en modo `P` y requieren conversión a escala de grises y redimensionado.
- Para CTC, el número de pasos de tiempo a la salida de la CNN debe superar con holgura la longitud máxima de etiqueta (75 tokens).

## Evaluación

- Las particiones por grupo reducen la fuga, pero no la eliminan por completo. Obras distintas con incipits muy parecidos pueden seguir en particiones diferentes.
- Las métricas por clase de los tokens raros tendrán poca validez estadística.

## Trabajo futuro (fuera de la primera versión)

- Interfaz web.
- Lectura de PDF multipágina.
- Partituras manuscritas.
- Evaluación en imágenes de Camera-PrIMuS.