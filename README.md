# TFG_Proyect
Trabajo final de grado

miguelangel@lcc.uma.es
lun, 2 feb, 15:01 (hace 3 días)
para mí, Paula

Buenas,

Veo que vas bien con los experimentos y que ya has pasado la fase inicial, que es la más difícil :)

Los resultados eran más o menos los esperados, así que no hay problema por ello.

Efectivamente, tal y como apuntas la dificultad de este tipo de problemas radica en el número tan grande de posibles configuraciones. Antes que nada, documenta todo lo que has probado para que el tribunal sea consciente de todos los experimentos que has realizado.

 

Obtén las métricas de las imágenes con un solo canal de cada una de las mejores binarizaciones y filtros morfológicos para detección según el artículo de Ignacio, que son: nc5, no2, sc2, so3.

Obtén las métricas de las imágenes con un solo canal de cada una de las mejores binarizaciones y filtros morfológicos para clasificación según el artículo de Ignacio, que son: nc2, no3, sc4, so5.

Obtén las métricas de las imágenes con un solo canal de cada una de las texturas: dissimilarity, mean, contrast, std, homogeneity, ASM, energy, entropy, max.

Haz las siguientes combinaciones (los mejores los determinamos según su F1 obtenido):

- Sin procesar, mejor binarización y filtro morfológico detección, mejor textura detección.

- Sin procesar, mejor binarización y filtro morfológico clasificación, mejor textura detección.

- Sin procesar, mejor binarización y filtro morfológico detección, mejor textura clasificación.

- Sin procesar, mejor binarización y filtro morfológico clasificación, mejor textura clasificación.

- 3 mejores binarizaciones y filtros morfológicos.

- 3 mejores texturas.

- 2 mejores binarizaciones y filtros morfológicos (detección o clasificación), y la mejor textura detección.

- 2 mejores binarizaciones y filtros morfológicos (detección o clasificación), y la mejor textura clasificación.

- Mejor binarización y filtro morfológico detección, y las 2 mejores texturas (detección o clasificación).

- Mejor binarización y filtro morfológico clasificación, y las 2 mejores texturas (detección o clasificación).

 

A ver qué resultados se obtienen. En las métricas, además de la media como ya has hecho, obtén también la desviación típica.

 

Dado que el periodo de solicitud para la defensa es del 9 al 20 de febrero y que aún no tienes los experimentos terminados ni un borrador de la memoria, veo muy difícil que puedas presentarte en esta convocatoria.

Ánimo, que vas bien :)

Saludos,