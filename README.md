# Machine Learning Supervisado 
## Clasificación del nivel de engagement en jugadores de videojuegos online 🎮
##### Fuente: 
Rabie El Kharoua. (2024). 🎮 Predict Online Gaming Behavior Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8742674. https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset  

**OBJETIVO:**  
Predecir el nivel **nivel de engagement** que tendrá un jugador teniendo en cuenta una serie de características dadas como las horas de juego empleadas, las sesiones iniciadas, tipo de juego, etc.  

**METODOLOGÍA:**  
Nos encontramos ante un **problema de clasificación** para el cual emplearemos algoritmos de clasificación no supervisada.

**MÉTRICA:**  
Nos centraremos en el balanced accuracy y recall medio dada las características del target a predecir.  

**IMPACTO DE NEGOCIO:**  
 - Ayudar a mejorar estrategias de retención de usuarios.
 - Previsión de abandonos.
 - Mejorar las estrategias de monetización.
 - Ayudar en la toma de decisiones para el diseño de videojuegos.

**Documentos**  
- main.ipynb : Jupiter Notebook con el contenido del proyecto y conclusiones.
- src : Directorio de carpetas donde se encuentran imágenes empleadas en el dataset, datos, modelo entrenados y scripts utilizados.
- engagement_predictor.py : script con código para una presentación del proyecto desplegada en Streamlit además de una demo de predicciones del modelo.

**Requerimientos**  
Para ejecutar la aplicaicón web con Streamlit:
1. Abrir terminal
2. Descomprimir carpeta src/model
3. Ejecutar desde el directorio donde se encuentre el script: streamlit run engagement_predictor.py
