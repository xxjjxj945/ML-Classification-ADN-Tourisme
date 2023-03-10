# import python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

# folium + streamlit
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu

# pickle pour mdl Machine Learning
import pickle

st.set_page_config(layout="wide")
st.title('Bienvenue chez _ADN tourisme_ 🗼')

# data !
@st.cache_data   
def get_my_data():    
    # chargement des donnees    
    data = pd.read_csv('df_21col_19_02_2023.csv')    
    return data

def get_my_model():
    # Charger le modèle
    with open('modelLR.pkl', 'rb') as file:
        new_modelLR = pickle.load(file)
    with open('tfidf.pkl', 'rb') as file:
        new_tfidf = pickle.load(file)
    return new_modelLR,new_tfidf

def isNaN(num):
    return num != num

# Recuperation des données+mdl
df = get_my_data()
new_modelLR, new_tfidf = get_my_model()

#########################################
# Sidebar
#########################################

image2 = Image.open("Tourisme-France.jpg")

# Nouvelle taille image
width2, height2 = image2.size
new_height2 = 250
new_width2 = int(width2 * new_height2 / height2)
resized_image2 = image2.resize((new_width2, new_height2))

# Sauvegarde et affichage image
resized_image2.save("wild-resized.jpg")
st.sidebar.image("wild-resized.jpg")
st.sidebar.title("Filtrer par Région et Département")

# Filtres sidebar
# Region, departements
selected_region = st.sidebar.selectbox("Sélectionner une Région", df["region"].drop_duplicates())
departement_liste =  df[df["region"] == selected_region]["departement"].drop_duplicates().to_list()
if selected_region=='Île-de-France':
    departement_liste.insert(0,'Paris')
selected_departement = st.sidebar.selectbox("Sélectionner un Département", departement_liste)

# Etablissements
options = df["type_etablissement_extraction"].drop_duplicates()
options = options[~options.isin(["NaN", "Autre"])] # remove unwanted values
options = ["All"] + options.tolist() # add "All" as the first option
selected_types = st.sidebar.multiselect("Sélectionner le type d'établissement", options, 'Restauration')

# Apply filters to data
if "All" in selected_types:
    selected_types = options[1:] # remove "All" from the selected options

# Menu horizontal
selected2 = option_menu(None, ["Accueil", "Détail", "📈Statistique", "💻Robot ML"], 
    icons=['house', 'list-task', ' ',' '], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#########################################
# Accueil
#########################################
if selected2=='Accueil':

    new_title = '''<p style="font-family:sans-serif  ; font-size: 20px;">Cartographie des points d'intérêts (POI)</p>'''
    st.markdown(new_title, unsafe_allow_html=True)
    
    # Apply filters to data
    filtered_df = df[(df["region"] == selected_region) & (df["departement"] == selected_departement)] 

    #centrage de la carte sur la premiere adresse
    latitude = filtered_df['lat'].mean()
    longitude = filtered_df['lng'].mean()
    point = [latitude, longitude]
    width, height = 800, 400

    fig = folium.Figure(width=width, height=height)

    m = folium.Map(location=point,zoom_start=11, control_scale=True, width=width, height=height)

    if not selected_types:
        st.warning('No type selected')
    else:
        for selected_type in selected_types:
            df_mask = filtered_df[filtered_df["type_etablissement_extraction"] == selected_type]
            df_mask = df_mask.reset_index()
            if df_mask.empty:
                st.warning(f'No data available for this type: {selected_type}')
                continue
            marker_cluster = MarkerCluster().add_to(m)

            for id in range(len(df_mask)):
                
                nom = df_mask.loc[id,'nom']
                point = [df_mask.loc[id,'lat'],df_mask.loc[id,'lng']]
                website = df_mask.loc[id,'url_proprietaire']
                tel = df_mask.loc[id,'tel__proprietaire']

                if not isNaN(df_mask.loc[id,'rue']) :
                    adresse = df_mask.loc[id,'rue']+' '+df_mask.loc[id,'localite']+' '+str(int(df_mask.loc[id,'postal_code']))
                else:
                    adresse = 0

                # Code html ds le pop-up quand on clique sur un etablissement
                code_html = f"""
                <p style="text-align: center;"><span style="font-family: Didot, serif; font-size: 18px;">{nom}</span></p>
                <p style="text-align: center;"><span style="font-family: Didot, serif; font-size: 16px;">{adresse}</span></p>
                """
                if not isNaN(tel):
                 code_html +=  f"""
                <p style="text-align: center;"><span style="font-family: Didot, serif; font-size: 16px;">{tel}</span></p>
                """
                if not isNaN(website):
                 code_html +=  f"""
                <p style="text-align: center;"><a href={website} target="_blank" title="Website"><span style="font-family: Didot, serif; font-size: 16px;">{nom}</span></a></p>
                """
                pub_html = folium.Html(code_html, script=True)
                
                # Create pop-up with html content
                popup = folium.Popup(pub_html, max_width=700)
                
                folium.Marker(location=point,popup=popup,tooltip=nom).add_to(marker_cluster)

    folium.LayerControl().add_to(m)

    fig.add_child(m)
    folium_static(fig)

#########################################
# Detail
#########################################
elif selected2=='Détail':

    selected_name = st.sidebar.selectbox("Sélectionner un nom", df[(df["departement"] == selected_departement) & (df["type_etablissement_extraction"] == selected_types[0])]["nom"].drop_duplicates())
    filtered_df = df[(df["nom"] == selected_name)]
    
    new_title = '''<p style="font-family:sans-serif  ; font-size: 20px;">Informations de l'établissement selectionné</p>'''
    st.markdown(new_title, unsafe_allow_html=True)
      
    # Dictionary des inforations clients
    client_info = {
        "Nom": filtered_df.loc[:, "nom"].values[0],
        "Type": filtered_df.loc[:, "type_etablissement_extraction"].values[0],
        "Descriptif": filtered_df.loc[:, "descriptif"].values[0],
        "Adresse": f"{filtered_df.loc[:, 'rue'].values[0]}, {filtered_df.loc[:, 'localite'].values[0]}, {filtered_df.loc[:, 'postal_code'].values[0]}",
        "Département": filtered_df.loc[:, "departement"].values[0],
        "Région": filtered_df.loc[:, "region"].values[0],
        "Tél": filtered_df.loc[:, "tel__proprietaire"].values[0],
        "Email": filtered_df.loc[:, "email_proprietaire"].values[0],
        "Code_insee": filtered_df.loc[:, "code_insee"].values[0],
        "Site Web adn tourisme": filtered_df.loc[:, "url_etablissment"].values[0],
        "Site Web etablissement": filtered_df.loc[:, "url_proprietaire"].values[0],
        "Nom éditeur": filtered_df.loc[:, "name_publisher"].values[0],
        "Courriel éditeur: ": filtered_df.loc[:, "courriel_publisher"].values[0],
        "Site Web éditeur: ": filtered_df.loc[:, "url_publisher"].values[0],
        "Maj client: ": filtered_df.loc[:, "maj_update"].values[0],
        "Maj datatourisme: ": filtered_df.loc[:, "maj_update_datatourisme"].values[0]}
    
    # Affichage des inforations clients
    for key, value in client_info.items():
        st.write(f"{key}: {value}")

#########################################
# Statistique
#########################################
elif selected2=='📈Statistique':

    new_title = '<p style="font-family:sans-serif  ; font-size: 20px;">Statistique</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # nbre POI
    nbre_POI = df.shape[0]
   
    # nbre de department, region
    region_couverte = df.region.drop_duplicates()
   
    # departement couvert
    departement_couvert = df.departement.drop_duplicates()

    # nbre POI departement
    departement = selected_departement
    nbre_POI_departement = df[df['departement']==departement].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('POI (Total): ', nbre_POI)
    col2.metric('Regions couvertes : ', len(region_couverte))
    col3.metric('Département couverts : ', len(departement_couvert))
    col4.metric(f'POI ({selected_departement})', nbre_POI_departement)
  
    col3_1, col3_2 = st.columns([1, 1])
    
    # Affichage par colonne :
    with col3_1:
        fig3 = plt.figure(figsize=(5,5))
        filtered_df = df[df["region"] == selected_region]
        region_counts = filtered_df.groupby("departement")["nom"].nunique()
        sorted_region_counts = region_counts.sort_values(ascending=False)

        sns.barplot(
            data=pd.DataFrame(sorted_region_counts,columns=['nom']), 
            x='nom',
            y=sorted_region_counts.index,
            palette='Blues_d')
        plt.title('Répartition des POI par département\n('+selected_region+')')
        plt.show()
        st.pyplot(fig3)

    with col3_2:
        # Repartition POI
        fig2 = plt.figure(figsize=(5,5))
        df_e = df[df["region"] == selected_region]['type_etablissement_extraction']
        idx = df_e.value_counts()[df_e.value_counts()<df_e.value_counts()[3]].index
        df_donut = df_e.apply(lambda x: x if x not in idx else 'Autres')
        my_circle = plt.Circle( (0,0), 0.7, color='white')
        df_donut.value_counts(normalize=True).plot(kind='pie',autopct = '%1.1f%%',pctdistance = 0.85, ylabel=None);
        p = plt.gcf()
        plt.title("Répartition du nombre d'établissements\n(" + selected_region  +')' )
        plt.ylabel('')
        p.gca().add_artist(my_circle)
        st.pyplot(fig2)

    col3_3, col3_4 = st.columns([1, 1])
    with col3_3:

        fig = plt.figure(figsize=(5,5))
        my_df = pd.DataFrame(df[df["region"] == selected_region].isna().mean().sort_values(ascending=False)*100)
        my_df.columns = ['Valeurs manquantes']
        my_df = my_df.drop(index=['Unnamed: 0'])
        sns.barplot(data=my_df[:10],
        x='Valeurs manquantes',
        y=my_df[:10].index,
        palette='dark:salmon_r')
        plt.title("Valeurs manquantes\n("+selected_region+')')
        st.pyplot(fig)

#########################################
# Exemple de classification ML
#########################################
elif selected2=='💻Robot ML':
    
    new_title = '<p style="font-family:sans-serif  ; font-size: 20px;">Recommandation de catégorie à partir du descriptif<br/><br/></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    exemple_txt = '''Aujourd’hui, notre chef vous propose une cuisine traditionnelle raffinée,  élaborée avec des produits 
frais, de saison dans un décor feutré, en plein cœur du 15ème arrondissement de Paris'''
    txt = st.text_area("Veuillez renseigner un descriptif", exemple_txt)

    result = new_modelLR.predict(new_tfidf .transform(pd.Series(txt)))[0]
    new_title = '<p style="font-family:sans-serif  ; font-size: 20px;">➡️'+result+'</p>'
    st.markdown(new_title, unsafe_allow_html=True)
  
