import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pickle
from time import time
from PIL import Image
from sklearn.model_selection import train_test_split, LearningCurveDisplay, learning_curve
from sklearn.metrics import ConfusionMatrixDisplay
from utils import bootcampviztools
import os


st.sidebar.title("Menu")
menu = st.sidebar.radio(' ',['Inicio','Análisis', 'Comparativa de modelos', 'Demo Predicción'])

if menu == 'Inicio':
    st.title('Predicción de engagement en Juegos Online :video_game:')
    st.header('')
    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.header(':dart:')
        st.header('Objetivo')
        st.markdown('Predecir el nivel de engagement de un jugador teniendo en cuenta una serie de características como:')
        st.markdown('- Inicios de sesion')
        st.markdown('- Tiempo de juego')
        st.markdown('- Logros conseguidos')
        st.markdown('- Edad, etc..')

    with col2:
       st.header(':pencil:')
       st.header('Metodo')
       st.markdown('ML Supervisado')
       st.markdown('Problema de clasificación multiclase')
       st.markdown('Métricas: Balanced Accuracy/ Recall Medio')
    
    with col3:
       st.header(':rocket:')
       st.header('Impacto')
       st.markdown('Mejorar estrategias de retención')
       st.markdown('Dectección de abandono')
       st.markdown('Mejoras en diseño y desarrollo')

elif menu == 'Análisis': 
    st.title('Visualización de datos y variable objetivo')

    st.header(':dart: Target')

    #importamos los datos del dataset para generar los sets de train y test para las gráficas
    #data = pd.read_csv("./data/raw/online_gaming_behavior_dataset.csv")
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    data= pd.read_csv(os.path.join(BASE_PATH, './data/raw/online_gaming_behavior_dataset.csv' ))
    target = 'EngagementLevel'
    train_set, test_set = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)

    #generamos datos para pintar la frecuencia relativa del target
    frec_abs = train_set[target].value_counts()
    frec_rel = (frec_abs/ frec_abs.sum())*100
    df_frec = frec_rel.reset_index()
    df_frec.columns = ['EngagementLevel', 'Frecuencia_Relativa']

    #grafica target:
    target_plot = px.bar(df_frec, x=target, y= 'Frecuencia_Relativa', title=f'Distribución de {target}',
                               color=target, text_auto=True, 
                               color_discrete_sequence= px.colors.cyclical.Twilight)
    
    target_plot.update_layout(xaxis_title='Nivel de Engagement',
                      yaxis_title='Frecuencia relativa (%)')
    
    st.plotly_chart(target_plot)

    #visualización del comportamiento de features numéricas con respecto al target
    st.header(':chart_with_upwards_trend: Comportamiento de features numéricas con respecto al target')


    #Age vs target
    plot_age =  px.histogram(train_set, x='Age', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Edad por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_age.update_layout(xaxis_title='Edad',
                      yaxis_title='Densidad')
                      #legend='EngagementLevel')
    
    st.plotly_chart(plot_age)

    #PlaytimeHours vs target
    plot_hours =  px.histogram(train_set, x='PlayTimeHours', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Horas de Juego por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_hours.update_layout(xaxis_title='Horas de Juego',
                      yaxis_title='Densidad')
                      #legend='EngagementLevel')
    
    st.plotly_chart(plot_hours)


    #Sesiones semana vs target
    plot_sessions =  px.histogram(train_set, x='SessionsPerWeek', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Sesiones/Semana por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_sessions.update_layout(xaxis_title='Sesiones a la semana',
                      yaxis_title='Densidad')
    
    st.plotly_chart(plot_sessions)


    #Minutos de juego a al semana vs target
    plot_minutes_week =  px.histogram(train_set, x='AvgSessionDurationMinutes', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Tiempo/sesion (minutos) por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_minutes_week.update_layout(xaxis_title='Tiempo/sesion (minutos)',
                      yaxis_title='Densidad')
    
    st.plotly_chart(plot_minutes_week)


    #Minutos de Nivel vs target
    plot_level =  px.histogram(train_set, x='PlayerLevel', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Nivel Jugador por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_level.update_layout(xaxis_title='Nivel del Jugador',
                      yaxis_title='Densidad')
    
    st.plotly_chart(plot_level)


    #Minutos de logros vs target
    plot_logros =  px.histogram(train_set, x='AchievementsUnlocked', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Distribución de Logros obtenidos por Nivel de Engagement',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_logros.update_layout(xaxis_title='Logros Obtenidos',
                      yaxis_title='Densidad')
    
    st.plotly_chart(plot_logros)

    #HeatMap
    num_col = [col for col in data.columns if data[col].dtype != 'object']
    num_col.remove('PlayerID') #nos deshacemos de ID al tener alta cardinalidad y actuar como índice ya que no aportará información
    num_col.remove('InGamePurchases') #metemos esta varibale como categórica, aunque ya está codificada
    
    heatmap = px.imshow(train_set[num_col].corr(), text_auto=True, width= 1200, height=800, aspect='auto',  
                        color_continuous_scale='magma', title='Matriz de correlación para variables numéricas')
    st.plotly_chart(heatmap)


    #visualizacion del comportamiento de features categoroicas con respecto al target
    st.header(':chart_with_upwards_trend: Comportamiento de features categóricas con respecto al target')

    #Genero vs target
    #Repetir operación de generar df con valores absolutos y frecuencias relativas
    train_set_df = pd.DataFrame(train_set)
    frec_rel_tg = train_set_df.groupby([target, 'Gender']).size().reset_index(name='Frecuencia_Absoluta')
    frec_abs_total = train_set_df.groupby(target).size().reset_index(name="Total")
    frec_rel_tg = frec_rel_tg.merge(frec_abs_total, on=target)
    frec_rel_tg["Frecuencia_Relativa"] = round((frec_rel_tg["Frecuencia_Absoluta"] / frec_rel_tg["Total"]) * 100, 2)


    plot_gender =  px.bar(frec_rel_tg, x=target, y='Frecuencia_Relativa', color='Gender', text_auto=True,
                            barmode='stack', opacity=0.9, title='Distribución de Género por Nivel de Engagement',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_gender.update_layout(xaxis_title=target,
                      yaxis_title='Frecuencia Relativa (%)')
    
    st.plotly_chart(plot_gender)


    #Ubicacion vs target
    frec_rel_tl = train_set_df.groupby([target, 'Location']).size().reset_index(name='Frecuencia_Absoluta')
    frec_rel_tl = frec_rel_tl.merge(frec_abs_total, on=target)
    frec_rel_tl["Frecuencia_Relativa"] = round((frec_rel_tl["Frecuencia_Absoluta"] / frec_rel_tl["Total"]) * 100, 2)


    plot_location =  px.bar(frec_rel_tl, x=target, y='Frecuencia_Relativa', color='Location', text_auto=True,
                            barmode='stack', opacity=0.9, title='Distribución de Ubicación por Nivel de Engagement',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_location.update_layout(xaxis_title=target,
                      yaxis_title='Frecuencia Relativa (%)')
    
    st.plotly_chart(plot_location)


    #Gnero de juego vs target
    frec_rel_tgg = train_set_df.groupby([target, 'GameGenre']).size().reset_index(name='Frecuencia_Absoluta')
    frec_rel_tgg = frec_rel_tgg.merge(frec_abs_total, on=target)
    frec_rel_tgg["Frecuencia_Relativa"] = round((frec_rel_tgg["Frecuencia_Absoluta"] / frec_rel_tgg["Total"]) * 100, 2)


    plot_game =  px.bar(frec_rel_tgg, x=target, y='Frecuencia_Relativa', color='GameGenre', text_auto=True,
                            barmode='stack', opacity=0.9, title='Distribución de Género de Juego por Nivel de Engagement',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_game.update_layout(xaxis_title=target,
                      yaxis_title='Frecuencia Relativa (%)')
    
    st.plotly_chart(plot_game)


    #Dificultad vs target
    frec_rel_tgd = train_set_df.groupby([target, 'GameDifficulty']).size().reset_index(name='Frecuencia_Absoluta')
    frec_rel_tgd = frec_rel_tgd.merge(frec_abs_total, on=target)
    frec_rel_tgd["Frecuencia_Relativa"] = round((frec_rel_tgd["Frecuencia_Absoluta"] / frec_rel_tgd["Total"]) * 100, 2)


    plot_dif =  px.bar(frec_rel_tgd, x=target, y='Frecuencia_Relativa', color='GameDifficulty', text_auto=True,
                            barmode='stack', opacity=0.9, title='Distribución de Dificultad del Juego por Nivel de Engagement',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_dif.update_layout(xaxis_title=target,
                      yaxis_title='Frecuencia Relativa (%)')
    
    st.plotly_chart(plot_dif)


    #Compras vs target
    frec_rel_tc = train_set_df.groupby([target, 'InGamePurchases']).size().reset_index(name='Frecuencia_Absoluta')
    frec_rel_tc = frec_rel_tc.merge(frec_abs_total, on=target)
    frec_rel_tc["Frecuencia_Relativa"] = round((frec_rel_tc["Frecuencia_Absoluta"] / frec_rel_tc["Total"]) * 100, 2)


    plot_compras =  px.bar(frec_rel_tc, x=target, y='Frecuencia_Relativa', color='InGamePurchases', text_auto=True,
                            barmode='stack', opacity=0.9, title='Distribución de Compras por Nivel de Engagement',
                            color_discrete_sequence= px.colors.cyclical.mrybm)
    
    plot_compras.update_layout(xaxis_title=target,
                      yaxis_title='Frecuencia Relativa (%)')
    
    st.plotly_chart(plot_compras)


elif menu == 'Comparativa de modelos': 
    st.title('Escogiendo el mejor modelo para clasificación')
    st.subheader(':dart: Controlar el overfitting y mejorar el rendimiento del modelo')
    st.markdown('')
    st.markdown('')


    st.header(':pushpin: Baseline') 
    st.markdown('Resultados de cross_validate combinado con StratifiedKFold()')
    #importamos df y añadimos columna con url de la imagen para el df interactivo
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    baseline_df = pd.read_csv(os.path.join(BASE_PATH,'./data/df_resultados_cv.csv'))
    baseline_df['img'] = ['./img/baseline_curves/b_dt_curve.png', './img/baseline_curves/b_rl_curve.png', 
                          './img/baseline_curves/b_rf_curve.png', './img/baseline_curves/b_xgb_curve.png',
                           './img/baseline_curves/b_lgbm_curve.png', './img/baseline_curves/b_knn_curve.png',
                            './img/baseline_curves/b_cb_curve.png' ]

    st.dataframe(baseline_df)

    model_selected = st.multiselect('Selecciona un modelo:', #creamos la multiselección en el dataframe
                                    options=baseline_df['modelo'], default=[])
    filas_seleccionadas = baseline_df[baseline_df['modelo'].isin(model_selected)] #filtro por filas seleccionadas

    if not filas_seleccionadas.empty:
        for imagen, fila in filas_seleccionadas.iterrows(): #iteramos sobre las filas para abrir las imagenes desde la ruta proporcionada
            imagen = Image.open(fila['img'])
            st.image(imagen)
    
    #Apartado de comparativa de modelos seleccionados
    st.subheader('')
    st.header(':computer: Comparativa de modelos')
    st.markdown('Resultados de optimización y evaluación')

    #Creamos desplegable para mostrar imágenes de curvas y matriz de confusión como en el apartado anterior
    opciones = ['LGBMClassifier', 'RandomForestClassifier', 'LogisticRegressor', 'MLP']
    desplegable = st.selectbox('Selecciona un modelo:', opciones)
   
   #iteramos para realizar las acciones correspondientes a cada acción
    if desplegable == 'LGBMClassifier':
        st.subheader('Resultados para LGBMClassifier')
        st.markdown('''A continuación se muestran los resultados de un modelo entrenado con sampling y feature selection
                     y otro modelo optimizado para corregir la tendencia al overfitting, sin sampling ni feature selección''')

    
        imagenes_lgbm = ['./img/lgbm_plots/lgbm_optimized_curve.png', './img/lgbm_plots/lgbm_conf_matrix_optimized.png',
                         './img/lgbm_plots/lgbm_optimized_1_curve.png', './img/lgbm_plots/lgbm_conf_matrix_optimized_1.png']
        
        for imagen in imagenes_lgbm:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparativa')
        #df_comp_lgbm = pd.read_csv('./data/df_comp_lgbm.csv')
        df_comp_lgbm= pd.read_csv(os.path.join(BASE_PATH, './data/df_comp_lgbm.csv' ))
        df_comp_lgbm.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_lgbm)


    if desplegable == 'RandomForestClassifier':
        st.subheader('Resultados para RandomForestClassifier')
        st.markdown('''A continuación se muestran las curvas de aprendizaje de un modelo entrenado con sampling y feature selection, 
                    otro modelo optimizado para corregir la tendencia al overfitting, sin sampling ni feature selección y por último un modelo 
                    entrenado para determinar en qué punto el modelo comienza a tender al sobreajuste.
                    ''')
    
        imagenes_rf = ['./img/rf_plots/rf_optimized_1_curve.png', './img/rf_plots/rf_optimized_2_curve.png', 
                         './img/rf_plots/rf_optimized_3_curve.png', './img/rf_plots/rf_conf_matrix_optimized_2.png']
        
        for imagen in imagenes_rf:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparativa')
        df_comp_rf = pd.read_csv(os.path.join(BASE_PATH,'./data/df_comp_rf.csv'))
        df_comp_rf.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_rf)


    if desplegable == 'LogisticRegressor':
        st.subheader('Resultados para LogisticRegressor')
        st.markdown('''A continuación se muestran los resultados de un modelo entrenado feature selection''')
    
        imagenes_lr = ['./img/logisticreg_plots/rl_optimized_curve.png','./img/logisticreg_plots/rl_conf_matrix_optimized.png']
        
        for imagen in imagenes_lr:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparativa')
        df_comp_lr = pd.read_csv(os.path.join(BASE_PATH,'./data/df_comp_lr.csv'))
        df_comp_lr.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_lr)


    if desplegable == 'MLP':
        st.subheader('Resultados para Red Neural Simple (MLP)')
        st.markdown('**Arquitectura:**')
        code = ''' 
                    modelo_1 = keras.Sequential()
                    modelo_1.add(keras.Input(shape=(X_train_dl.shape[1],))) 
                    modelo_1.add(keras.layers.Dense(50, activation='relu'))
                    modelo_1.add(keras.layers.Dense(len(np.unique(y_train_encoded)), 
                    activation='softmax'))
                    '''
        st.code(code, language='python')
        img_mlp = './img/Arquitectura_modelo_mlp.png'
        st.image(img_mlp, use_container_width=True)

        st.markdown('**Métricas:**')
    
        imagenes_dl = ['./img/dl_plots/acc_val_acc_curve_dl.png','./img/dl_plots/loss_val_loss_curve_dl.png', 
                       './img/dl_plots/dl_conf_matrix.png']
        
        for imagen in imagenes_dl:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)


#preparamos la pagina de la demo
elif menu == 'Demo Predicción': 
    st.title('Inserta parámetros para que el modelo realice una predicción')
    st.markdown('')

    #importamos modelo seleccionado y preprocesador
    with open ('./model/preprocessing.pkl', 'rb') as file:
        preprocessing = pickle.load(file)
    
    with open('./model/Modelo_LGBMClassifier_Controled', 'rb') as file:
        unpickle_model_lgbm_controled = pickle.load(file)

    st.subheader('**:birthday: Edad**')
    edad = st.slider('Inserta edad', 10,50,1)
    st.write('Edad seleccionada:', edad, 'años')

    st.subheader('**:couple: Género**')
    genero = st.selectbox('Selecciona el género:', options=['Male','Female'])
    st.write('Género seleccionado:', genero)

    st.subheader('**:airplane: Ubicacion**')
    location = st.selectbox('Selecciona ubicación:', options=['Other', 'USA', 'Europe', 'Asia'])
    st.write('Ubicación seleccionada:', location)

    st.subheader('**:video_game: Género de Juego**')
    genre= st.select_slider('Selecciona tipo de juego:', options=['Action', 'RPG', 'Strategy', 'Sports', 'Simulation'])
    st.write('Juego seleccionado:', genre)

    st.subheader('**:clock10: Horas de Juego**')
    hours = st.number_input('Inserta un número de 0 a 50', min_value=0, max_value=50, step=1, value=10)
    st.write('Has insertado', hours, 'horas')

    st.subheader('**:money_with_wings: Compras**')
    compras = st.number_input('Inserta 0: No o 1: Sí', min_value=0, max_value=1, step=1, value=1)
    st.write('Has insertado', compras)

    st.subheader('**:game_die: Dificultad del Juego**')
    dific = st.selectbox('Elige un nivel de dificultad:', options=['Easy', 'Hard', 'Medium'])
    st.write('Nivel seleccionado:', dific)

    st.subheader('**:computer: Inicios de sesión por semana**')
    sessions = st.slider('Inserta número de inicios de sesión', 0,30,2)
    st.write('Número de inicios de sesión por semana:', sessions)


    st.subheader('**:clock10: Minutos de juego por sesión**')
    mins = st.slider('Inserta tiempo de juego', 0,200, 5)
    st.write('Minutos de juego seleccionados:', mins)

    st.subheader('**:trophy: Nivel**')
    nivel = st.slider('Inserta nivel', 0,100, 5)
    st.write('Has seleccionado nivel', nivel)


    st.subheader('**:trophy: Logros Conseguidos**')
    logros = st.slider('Inserta cantidad de logros', 0,100, 5)
    st.write('Logros seleccionados:', logros)

    #creamos un dataframe con los nuevos datos
    nuevos_datos = pd.DataFrame({'Age' : [edad],'Gender': [genero],
                                 'Location' : [location], 'GameGenre': [genre],
                                 'PlayTimeHours' : [hours], 'InGamePurchases': [compras],
                                 'GameDifficulty' : [dific], 'SessionsPerWeek' : [sessions],
                                 'AvgSessionDurationMinutes': [mins], 'PlayerLevel' : [nivel],
                                 'AchievementsUnlocked' : [logros]})
    
    #predicciones
    st.header('')
    if st.button('Predecir'):
        texto = 'Realizando predicción, ten paciencia :relaxed:'
        barra_progreso = st.progress(0, text=texto)

        for progreso in range(100):
            barra_progreso.progress(progreso + 1, text=texto)
    
        pred = unpickle_model_lgbm_controled.predict(nuevos_datos) #no es necesario el preprocesado previo, 
                                                                    #lo hace el propio modelo entrenado con pipeline
        labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        pred_label = [labels[prediccion] for prediccion in pred]
        st.success('Prediccón realizada con éxito')
        st.header('Nivel de engagement:')
        st.subheader(pred_label)
        





            


    




    

