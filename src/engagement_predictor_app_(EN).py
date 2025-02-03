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
menu = st.sidebar.radio(' ',['Home','Analysis', 'Model Comparative', 'Prediction Demo'])

if menu == 'Home':
    st.title('Enagagement Level prediction in online games :video_game:')
    st.header('')
    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.header(':dart:')
        st.header('Objectives')
        st.markdown("Predict a player's level of engagement by taking into account a number of characteristics such as:")
        st.markdown('- Logins')
        st.markdown('- Game time')
        st.markdown('- Achievements')
        st.markdown('- Age, etc..')

    with col2:
       st.header(':pencil:')
       st.header('Method')
       st.markdown('Supervised ML')
       st.markdown('Multiclass classification problem')
       st.markdown('Metrics: Balanced Accuracy/ Macro Recall')
    
    with col3:
       st.header(':rocket:')
       st.header('Bussines Impact')
       st.markdown('Improving retention strategies')
       st.markdown('Dropout Detection')
       st.markdown('Design and Development improvements')

elif menu == 'Analysis': 
    st.title('Data visualization and target')

    st.header(':dart: Target')

    #import dataset and train, test split
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    data= pd.read_csv(os.path.join(BASE_PATH, './data/raw/online_gaming_behavior_dataset.csv' ))
    target = 'EngagementLevel'
    train_set, test_set = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)

    #data for relative frequence
    frec_abs = train_set[target].value_counts()
    frec_rel = (frec_abs/ frec_abs.sum())*100
    df_frec = frec_rel.reset_index()
    df_frec.columns = ['EngagementLevel', 'Relative_Frequency']

    #target graph:
    target_plot = px.bar(df_frec, x=target, y= 'Relative_Frequency', title=f'{target} distribution',
                               color=target, text_auto=True, 
                               color_discrete_sequence= px.colors.cyclical.Twilight)
    
    target_plot.update_layout(xaxis_title='Engagement Level',
                      yaxis_title='Relative frequence (%)')
    
    st.plotly_chart(target_plot)

    #Numerical variables behabiour in relation with the target
    st.header(':chart_with_upwards_trend: Numerical variables behabiour in relation with the target')


    #Age vs target
    plot_age =  px.histogram(train_set, x='Age', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Age distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_age.update_layout(xaxis_title='Age',
                      yaxis_title='Density')
                      #legend='EngagementLevel')
    
    st.plotly_chart(plot_age)

    #PlaytimeHours vs target
    plot_hours =  px.histogram(train_set, x='PlayTimeHours', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Play time hours distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_hours.update_layout(xaxis_title='Play time Hours',
                      yaxis_title='Density')
                      #legend='EngagementLevel')
    
    st.plotly_chart(plot_hours)


    #Weekly sessions vs target
    plot_sessions =  px.histogram(train_set, x='SessionsPerWeek', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Weekly sessions distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_sessions.update_layout(xaxis_title='Sesssions per week',
                      yaxis_title='Density')
    
    st.plotly_chart(plot_sessions)


    #Playing time in minutes vs target
    plot_minutes_week =  px.histogram(train_set, x='AvgSessionDurationMinutes', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Session durantion minutes distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_minutes_week.update_layout(xaxis_title='Session duration in minutes',
                      yaxis_title='Density')
    
    st.plotly_chart(plot_minutes_week)


    #Player level vs target
    plot_level =  px.histogram(train_set, x='PlayerLevel', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Player level distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_level.update_layout(xaxis_title='Player Level',
                      yaxis_title='Density')
    
    st.plotly_chart(plot_level)


    #Achievements vs target
    plot_logros =  px.histogram(train_set, x='AchievementsUnlocked', color=target, marginal='box',
                             opacity=0.6, histnorm='density', barmode='overlay', title='Achievements distribution per Engagement Level',
                             color_discrete_sequence= px.colors.sequential.Sunsetdark)
    
    plot_logros.update_layout(xaxis_title='Achievements Unlocked',
                      yaxis_title='Density')
    
    st.plotly_chart(plot_logros)

    #HeatMap
    num_col = [col for col in data.columns if data[col].dtype != 'object']
    num_col.remove('PlayerID') #remove this variable, high cardinality
    num_col.remove('InGamePurchases') #this is a categorical variable already codified
    
    heatmap = px.imshow(train_set[num_col].corr(), text_auto=True, width= 1200, height=800, aspect='auto',  
                        color_continuous_scale='magma', title='Numerical variables Matrix Correlation')
    st.plotly_chart(heatmap)


    #Categorical variables behabiour in relation with target
    st.header(':chart_with_upwards_trend: Categorical variables behabiour in relation with target')

    #Gender vs target
    #Get relative frequencies
    train_set_df = pd.DataFrame(train_set)
    frec_rel_tg = train_set_df.groupby([target, 'Gender']).size().reset_index(name='Absolut_Frequency')
    frec_abs_total = train_set_df.groupby(target).size().reset_index(name="Total")
    frec_rel_tg = frec_rel_tg.merge(frec_abs_total, on=target)
    frec_rel_tg["Relative_Frequency"] = round((frec_rel_tg["Absolut_Frequency"] / frec_rel_tg["Total"]) * 100, 2)


    plot_gender =  px.bar(frec_rel_tg, x=target, y='Relative_Frequency', color='Gender', text_auto=True,
                            barmode='stack', opacity=0.9, title='Gender distribution per Engagement Level',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_gender.update_layout(xaxis_title=target,
                      yaxis_title='Relative Frequency (%)')
    
    st.plotly_chart(plot_gender)


    #Location vs target
    frec_rel_tl = train_set_df.groupby([target, 'Location']).size().reset_index(name='Absolut_Frequency')
    frec_rel_tl = frec_rel_tl.merge(frec_abs_total, on=target)
    frec_rel_tl["Relative_Frecuency"] = round((frec_rel_tl["Absolut_Frequency"] / frec_rel_tl["Total"]) * 100, 2)


    plot_location =  px.bar(frec_rel_tl, x=target, y='Relative_Frecuency', color='Location', text_auto=True,
                            barmode='stack', opacity=0.9, title='Location distribution per Engagement Level',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_location.update_layout(xaxis_title=target,
                      yaxis_title='Relative Frequency (%)')
    
    st.plotly_chart(plot_location)


    #Game genre vs target
    frec_rel_tgg = train_set_df.groupby([target, 'GameGenre']).size().reset_index(name='Absolut_Frequency')
    frec_rel_tgg = frec_rel_tgg.merge(frec_abs_total, on=target)
    frec_rel_tgg["Relative_Frecuency"] = round((frec_rel_tgg["Absolut_Frequency"] / frec_rel_tgg["Total"]) * 100, 2)


    plot_game =  px.bar(frec_rel_tgg, x=target, y='Relative_Frecuency', color='GameGenre', text_auto=True,
                            barmode='stack', opacity=0.9, title='Game genre distribution per Engagement Level',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_game.update_layout(xaxis_title=target,
                      yaxis_title='Relative Frequency (%)')
    
    st.plotly_chart(plot_game)


    #Dificultad vs target
    frec_rel_tgd = train_set_df.groupby([target, 'GameDifficulty']).size().reset_index(name='Absolut_Frequency')
    frec_rel_tgd = frec_rel_tgd.merge(frec_abs_total, on=target)
    frec_rel_tgd["Relative_Frecuency"] = round((frec_rel_tgd["Absolut_Frequency"] / frec_rel_tgd["Total"]) * 100, 2)


    plot_dif =  px.bar(frec_rel_tgd, x=target, y='Relative_Frecuency', color='GameDifficulty', text_auto=True,
                            barmode='stack', opacity=0.9, title='Game difficulty distribution per Engagement Level',
                            color_discrete_sequence= px.colors.cyclical.mygbm)
    
    plot_dif.update_layout(xaxis_title=target,
                      yaxis_title='Relative Frequency (%)')
    
    st.plotly_chart(plot_dif)


    #Compras vs target
    frec_rel_tc = train_set_df.groupby([target, 'InGamePurchases']).size().reset_index(name='Absolut_Frequency')
    frec_rel_tc = frec_rel_tc.merge(frec_abs_total, on=target)
    frec_rel_tc["Relative_Frecuency"] = round((frec_rel_tc["Absolut_Frequency"] / frec_rel_tc["Total"]) * 100, 2)


    plot_compras =  px.bar(frec_rel_tc, x=target, y='Relative_Frecuency', color='InGamePurchases', text_auto=True,
                            barmode='stack', opacity=0.9, title='In game purchases per Engagement Level',
                            color_discrete_sequence= px.colors.cyclical.mrybm)
    
    plot_compras.update_layout(xaxis_title=target,
                      yaxis_title='Relative Frequency (%)')
    
    st.plotly_chart(plot_compras)


elif menu == 'Model Comparative': 
    st.title('Choosing the best model for classification')
    st.subheader(':dart: Controlling overfitting and improving model performance')
    st.markdown('')
    st.markdown('')


    st.header(':pushpin: Baseline') 
    st.markdown('Results of cross_validate combined with StratifiedKFold()')

    #import dataframe and add url column for dinamic see¡lection
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    baseline_df = pd.read_csv(os.path.join(BASE_PATH,'./data/df_resultados_cv.csv'))
    rutas_img = ['./img/baseline_curves/b_dt_curve.png', './img/baseline_curves/b_rl_curve.png', 
                          './img/baseline_curves/b_rf_curve.png', './img/baseline_curves/b_xgb_curve.png',
                           './img/baseline_curves/b_lgbm_curve.png', './img/baseline_curves/b_knn_curve.png',
                            './img/baseline_curves/b_cb_curve.png' ]
    
    baseline_df['img'] = [os.path.join(BASE_PATH, ruta) for ruta in rutas_img]

    st.dataframe(baseline_df)

    model_selected = st.multiselect('Choose a model:', #multiselection
                                    options=baseline_df['modelo'], default=[])
    filas_seleccionadas = baseline_df[baseline_df['modelo'].isin(model_selected)] #filter by row

    if not filas_seleccionadas.empty:
        for imagen, fila in filas_seleccionadas.iterrows(): 
            imagen = Image.open(fila['img'])
            st.image(imagen)
    
    #Section of model compatative results
    st.subheader('')
    st.header(':computer: Model Comparative')
    st.markdown('Optimization and evaluation results')

    #model drop-down
    opciones = ['LGBMClassifier', 'RandomForestClassifier', 'LogisticRegressor', 'MLP']
    desplegable = st.selectbox('Choose a model:', opciones)
   
   
    if desplegable == 'LGBMClassifier':
        st.subheader('LGBMClassifier results')
        st.markdown('''The results belongs to a model trained with sampling method and feature selection. The second result is
                    belongs to another model optimized with the aim to face the overfitting, without apply sampling methor or feature
                    selection''')

    
        imagenes_lgbm = ['./img/lgbm_plots/lgbm_optimized_curve.png', './img/lgbm_plots/lgbm_conf_matrix_optimized.png',
                         './img/lgbm_plots/lgbm_optimized_1_curve.png', './img/lgbm_plots/lgbm_conf_matrix_optimized_1.png']

        path_img_lgbm = [os.path.join(BASE_PATH, ruta) for ruta in imagenes_lgbm]

        
        for imagen in path_img_lgbm:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparative')
        #df_comp_lgbm = pd.read_csv('./data/df_comp_lgbm.csv')
        df_comp_lgbm= pd.read_csv(os.path.join(BASE_PATH, './data/df_comp_lgbm.csv' ))
        df_comp_lgbm.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_lgbm)


    if desplegable == 'RandomForestClassifier':
        st.subheader('RandomForestClassifier results')
        st.markdown('''Firstly we see the learning curve of a model trained with 
                    sampling and feature selection, secondly we see the learning curve of another model 
                    optimised to control overfitting and finally the curve of a third model to determine at what point 
                    overfitting starts to occur.
                    ''')
    
        imagenes_rf = ['./img/rf_plots/rf_optimized_1_curve.png', './img/rf_plots/rf_optimized_2_curve.png', 
                         './img/rf_plots/rf_optimized_3_curve.png', './img/rf_plots/rf_conf_matrix_optimized_2.png']

        path_img_rf = [os.path.join(BASE_PATH, ruta) for ruta in imagenes_rf]

        
        for imagen in path_img_rf:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparative')
        df_comp_rf = pd.read_csv(os.path.join(BASE_PATH,'./data/df_comp_rf.csv'))
        df_comp_rf.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_rf)


    if desplegable == 'LogisticRegressor':
        st.subheader('LogisticRegressor results')
        st.markdown('''Results of the model trained with feature selection''')
    
        imagenes_lr = ['./img/logisticreg_plots/rl_optimized_curve.png','./img/logisticreg_plots/rl_conf_matrix_optimized.png']

        path_img_lr = [os.path.join(BASE_PATH, ruta) for ruta in imagenes_lr]

        for imagen in path_img_lr:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)
            

        st.subheader('Comparative')
        df_comp_lr = pd.read_csv(os.path.join(BASE_PATH,'./data/df_comp_lr.csv'))
        df_comp_lr.rename(columns={'Unnamed: 0' : 'model'}, inplace=True)
        st.dataframe(df_comp_lr)


    if desplegable == 'MLP':
        st.subheader('Simple Neuronal Network (MLP)')
        st.markdown('**Arquitecture:**')
        code = ''' 
                    modelo_1 = keras.Sequential()
                    modelo_1.add(keras.Input(shape=(X_train_dl.shape[1],))) 
                    modelo_1.add(keras.layers.Dense(50, activation='relu'))
                    modelo_1.add(keras.layers.Dense(len(np.unique(y_train_encoded)), 
                    activation='softmax'))
                    '''
        st.code(code, language='python')
        img_mlp = os.path.join(BASE_PATH, './img/Arquitectura_modelo_mlp.png')
        st.image(img_mlp, use_container_width=True)

        st.markdown('**Metrics:**')
    
        imagenes_dl = ['./img/dl_plots/acc_val_acc_curve_dl.png','./img/dl_plots/loss_val_loss_curve_dl.png', 
                       './img/dl_plots/dl_conf_matrix.png']
        
        path_img_dl = [os.path.join(BASE_PATH, ruta) for ruta in imagenes_dl]
        
        for imagen in path_img_dl:
            img = Image.open(imagen)
            st.image(img, use_container_width=True)


#preparamos la pagina de la demo
elif menu == 'Demo Prediction': 
    st.title('Insert the parameters for model prediction')
    st.markdown('')
    
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    #importamos modelo seleccionado y preprocesador
    path_preprocessing = os.path.join(BASE_PATH, './model/preprocessing.pkl')
    with open (path_preprocessing, 'rb') as file:
        preprocessing = pickle.load(file)
    
    path_model =  os.path.join(BASE_PATH,'./model/Modelo_LGBMClassifier_Controled')
    with open(path_model, 'rb') as file:
        unpickle_model_lgbm_controled = pickle.load(file)

    st.subheader('**:birthday: Age**')
    edad = st.slider('Insert age', 10,50,1)
    st.write('Age selected:', edad, 'years')

    st.subheader('**:couple: Gender**')
    genero = st.selectbox('Select gender:', options=['Male','Female'])
    st.write('Selected gender:', genero)

    st.subheader('**:airplane: Location**')
    location = st.selectbox('Select the Location:', options=['Other', 'USA', 'Europe', 'Asia'])
    st.write('Selected location:', location)

    st.subheader('**:video_game: Game Genre**')
    genre= st.select_slider('Select the game genre:', options=['Action', 'RPG', 'Strategy', 'Sports', 'Simulation'])
    st.write('Selected game:', genre)

    st.subheader('**:clock10: Play time hours**')
    hours = st.number_input('Insert a number from 0 to 50', min_value=0, max_value=50, step=1, value=10)
    st.write('You choose', hours, 'hours')

    st.subheader('**:money_with_wings: Purchases**')
    compras = st.number_input('Insert 0: No or 1: Yes', min_value=0, max_value=1, step=1, value=1)
    st.write('You choose', compras)

    st.subheader('**:game_die: Game difficulty**')
    dific = st.selectbox('Choose a game difficulty level:', options=['Easy', 'Hard', 'Medium'])
    st.write('Selected level:', dific)

    st.subheader('**:computer: Logins per week**')
    sessions = st.slider('Insert a number', 0,30,2)
    st.write('Number of weekely logins:', sessions)


    st.subheader('**:clock10: Session duration in minutes**')
    mins = st.slider('Insert time', 0,200, 5)
    st.write('Total minutes selected:', mins)

    st.subheader('**:trophy: Player Level**')
    nivel = st.slider('Insert level', 0,100, 5)
    st.write('Slected level', nivel)


    st.subheader('**:trophy: Achievements Unlocked**')
    logros = st.slider('Insert an achievements quantity', 0,100, 5)
    st.write('Total of achievements:', logros)

    #creamos un dataframe con los nuevos datos
    nuevos_datos = pd.DataFrame({'Age' : [edad],'Gender': [genero],
                                 'Location' : [location], 'GameGenre': [genre],
                                 'PlayTimeHours' : [hours], 'InGamePurchases': [compras],
                                 'GameDifficulty' : [dific], 'SessionsPerWeek' : [sessions],
                                 'AvgSessionDurationMinutes': [mins], 'PlayerLevel' : [nivel],
                                 'AchievementsUnlocked' : [logros]})
    
    #predicciones
    st.header('')
    if st.button('Predict'):
        texto = 'Making prediction, be patience :relaxed:'
        barra_progreso = st.progress(0, text=texto)

        for progreso in range(100):
            barra_progreso.progress(progreso + 1, text=texto)
    
        pred = unpickle_model_lgbm_controled.predict(nuevos_datos) #no es necesario el preprocesado previo, 
                                                                    #lo hace el propio modelo entrenado con pipeline
        labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        pred_label = [labels[prediccion] for prediccion in pred]
        st.success('Successful prediction')
        st.header('Engagement Level:')
        st.subheader(pred_label)
        





            


    




    

