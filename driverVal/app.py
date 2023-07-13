import streamlit as st
import pandas as pd
import numpy as np


from PIL import Image, ImageTransform as transform
# st.image(Image.open('maws-banner.png'))
st.markdown('<style>h1{color:dark-grey;font-size:62px}</style>',unsafe_allow_html=True)
# st.sidebar.image(Image.open('maws-menu.png'))
# menu = ['Peta','Monitoring Nasional','Analisis Risiko Pemerintah Daerah','Tren & Histori Sentimen Publik', 'Sentimen Publik Terkini','Analisis LHKPN','Smart Monitoring Program Daerah']
menu = ['Pickups Analysis','Driver Verification','Family Location History']
choice = st.sidebar.selectbox("Choose Menu",menu)

if choice == 'Driver Verification':
    import numpy as np 
    from PIL import Image
    from tensorflow.keras.preprocessing import image

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    from keras.applications.vgg16 import VGG16
    from sklearn.metrics.pairwise import cosine_similarity
    vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(224, 224,3))
    for model_layer in vgg16.layers:
        model_layer.trainable = False

    # def load_image(image_path):
    #     # input_image = Image.open(image_path)
    #     # np_image = Image.open(image_path)
    #     input_image = np.array(image_path).astype('float32') / 255
    #     # input_image = transform.resize(np_image, (224, 224, 3))
    #     resized_image = input_image.resize((224, 224))
    #     return resized_image
    def load_image1(img_path):
        input_image = image.load_img(img_path, target_size=(224, 224))
        return input_image
        # resized_image = transform.resize(input_image, (224, 224, 3))
        # resized_image = input_image.resize((224, 224))
        # return resized_image 
    
    def load_image2(image_data):
        input_image = Image.open(image_data)
        resized_image = input_image.resize((224, 224))
        return resized_image
    def get_image_embeddings(object_image : image):
        image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
        image_embedding = vgg16.predict(image_array)
        return image_embedding
    
    def get_similarity_score(first_image : str, second_image : str):
        first_image = load_image1(first_image)
        second_image = load_image1(second_image)
        first_image_vector = get_image_embeddings(first_image)
        second_image_vector = get_image_embeddings(second_image)
        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
        return similarity_score

    # st.subheader('Peta Risiko Korupsi Pemerintah Daerah')
    st.subheader('Perhitungan & Perbandingan Index Luminosity')
    drivers = pd.read_csv('driverVal/driverprofile.csv')
    listdriver =  drivers['driver'].unique().tolist()
    ctr = st.selectbox("Pilih Pemda",listdriver)
    url1 = f'driverVal/{ctr}.png'
    img1 = Image.open(url1)
    st.image(img1)
    # data1 = st.file_uploader('')
    data1 = st.camera_input('')
    if data1 != None:
        img2 = load_image1(data1)
        st.image(img2)
        similarity_score = get_similarity_score(url1, data1)
        # st.write(f'Similarity score: {similarity_score}')
        st.write(f'Similarity score: {similarity_score[0]:.2f}')
    else:
        st.write('Please Capture Photo')
elif choice == 'Pickups Analysis':
    from streamlit_cropper import st_cropper
    from PIL import Image
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Upload an image and set some options for demo purposes
    st.header("Cropper Demo")
    # img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    img_file = st.camera_input('')
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    # aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    # aspect_dict = {
    #     "1:1": (1, 1),
    #     "16:9": (16, 9),
    #     "4:3": (4, 3),
    #     "2:3": (2, 3),
    #     "Free": None
    # }
    # aspect_ratio = aspect_dict[aspect_choice]
    aspect_ratio = (1, 1)

    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                    aspect_ratio=aspect_ratio)
        
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((224,224))
        st.image(cropped_img)
elif choice == 'Family Location History':
    # st.subheader('Peta Risiko Korupsi Pemerintah Daerah')
    DATE_COLUMN = 'date/time'
    # DATE_COLUMN = 'Time'
    DATA_URL = 'driverVal/history.csv'
    @st.cache_data
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        # lowercase = lambda x: str(x).lower()
        # data.rename(lowercase, axis='columns', inplace=True)
        # data = pd.read_csv(DATA_URL,sep=";",nrows=nrows)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
    # data.to_csv('data.csv',index=False)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Done!(using st.cache_data)")


    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)


    st.subheader('Number of pickups by hour')

    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

    st.bar_chart(hist_values)


    hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f'Map of all pickups at {hour_to_filter}:00')
    st.map(filtered_data)

    from pynetwork import draw_network
    import plotly.io as pio
    pio.templates.default = "plotly_dark"
    dfnet = pd.read_csv('driverVal/family.csv')
    # st.dataframe(dfnet, use_container_width=True)
    # parent = st.selectbox("Parent",dfnet['provinsi'].unique())
    # provinsi = st.selectbox("Children",dfnet['provinsi'].unique())
    # dfnet = pd.read_csv('')
    # dfnet = dfnet[(dfnet['type']=='Father')| (dfnet['type']=='Mother')]
    net1 = draw_network(dfnet,'attr','name','Burg', 'teal','nameid')
    net1.update_layout(title_text='Family linked')
    # net2 = draw_network(dfnet,'PARTAI','NAMA','Burg', 'teal','nilaitransaksi')
    st.plotly_chart(net1,use_container_width=True)
    st.dataframe(dfnet)
