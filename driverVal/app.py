import streamlit as st
import pandas as pd
import numpy as np


from PIL import Image, ImageTransform as transform
import streamlit.components.v1 as components
st.image(Image.open('driverVal/icons/banner.png'))
st.markdown('<style>h1{color:dark-grey;font-size:62px}</style>',unsafe_allow_html=True)
st.sidebar.image(Image.open('driverVal/icons/SFLS.png'))
menu = ['Family Linked','Driver Challenges','Safe & Fun Travel']
choice = st.sidebar.selectbox("Choose Menu",menu)

if choice == 'Family Linked':
    submenu1 = st.sidebar.radio("",('Family Live Tracking', 'History', 'Subscription'))
    if submenu1 == 'Family Live Tracking':
        components.html('''
                        <div class='tableauPlaceholder' id='viz1689438795014' style='position: relative'><noscript><a href='#'><img alt='Children&#39;s Route ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ch&#47;ChildrensRouteMonitor&#47;ChildrensRoute&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ChildrensRouteMonitor&#47;ChildrensRoute' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ch&#47;ChildrensRouteMonitor&#47;ChildrensRoute&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1689438795014');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
                        ''',
                        width=350,height=700)
    if submenu1 == 'History':
        DATE_COLUMN = 'date/time'
        DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
                'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

        @st.cache_data
        def load_data(nrows):
            data = pd.read_csv(DATA_URL, nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis='columns', inplace=True)
            data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
            return data



        # Create a text element and let the reader know the data is loading.
        data_load_state = st.text('Loading data...')
        # Load 10,000 rows of data into the dataframe.
        data = load_data(1000)
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

    # from pynetwork import draw_network
    # import plotly.io as pio
    # pio.templates.default = "plotly_dark"
    # dfnet = pd.read_csv('driverVal/family.csv')
    # # st.dataframe(dfnet, use_container_width=True)
    # # parent = st.selectbox("Parent",dfnet['provinsi'].unique())
    # # provinsi = st.selectbox("Children",dfnet['provinsi'].unique())
    # # dfnet = pd.read_csv('')
    # # dfnet = dfnet[(dfnet['type']=='Father')| (dfnet['type']=='Mother')]
    # net1 = draw_network(dfnet,'attr','name','Burg', 'teal','nameid')
    # net1.update_layout(title_text='Family linked')
    # # net2 = draw_network(dfnet,'PARTAI','NAMA','Burg', 'teal','nilaitransaksi')
    # st.plotly_chart(net1,use_container_width=True)
    # st.dataframe(dfnet)
elif choice == 'Driver Challenges':
    linke = 'https://degaya.mofdac.com/'

    components.iframe(linke, scrolling=True, height=1600)
    # components.html('''
    #         <div class='tableauPlaceholder' id='viz1683812683355' style='position: relative'><noscript><a href='#'><img alt='Tingkat Kerawanan dan Upaya Pencegahan Korupsi ' src='{linke}' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value={linke} /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndeksSPI&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndeksSPI&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1683812683355');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='977px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    #         ''',width=900,height=700)
elif choice == 'Safe & Fun Travel':
    submenu3 = st.sidebar.radio("",('Driver Verification', 'AI Driver Matching', 'Where to go?'))
    if submenu3 == 'Driver Verification':
        from streamlit_cropper import st_cropper
        import numpy as np 
        from PIL import Image
        from tensorflow.keras.preprocessing import image
        from keras.applications.vgg16 import VGG16
        from sklearn.metrics.pairwise import cosine_similarity
        from PIL import Image
        import io

        st.set_option('deprecation.showfileUploaderEncoding', False)
        # st.header("Capture the driver face")
        # img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
        
        
        vgg16 = VGG16(weights='imagenet', include_top=False, 
                pooling='max', input_shape=(224, 224,3))
        for model_layer in vgg16.layers:
            model_layer.trainable = False

        def load_image1(img_path):
            input_image = image.load_img(img_path, target_size=(224, 224))
            # input_image  = image.img_to_array(img_path)
            return input_image
        
        def load_image2(im):
            byteIO = io.BytesIO()
            im.save(byteIO, format='PNG')
            byteArr = byteIO.getvalue()
            return byteArr
            # input_image = Image.open(image_data)
            # resized_image = input_image.resize((224, 224))
            # return resized_image
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
        st.subheader('Computer Vision - Face Verification')
        drivers = pd.read_csv('driverVal/driverprofile.csv')
        listdriver =  drivers['driver'].unique().tolist()
        ctr = st.selectbox("Selected Driver",listdriver)
        url1 = f'driverVal/face/{ctr}.jpg'
        img1 = Image.open(url1)
        img_file = st.camera_input('')

        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                        aspect_ratio=(2, 2))
        
            # st.write("Preview")
            # _ = cropped_img.thumbnail((224,224))
            # st.image(cropped_img)
            similarity_score = get_similarity_score(url1,img_file)
            c1,c2 = st.columns((1,1))
            with c1:
                st.write("Driver Photo from database")
                img1 = img1.resize((224,224))
                st.image(img1)
            with c2:
                st.write("Driver Photo Captured")
                # _ = cropped_img.thumbnail((224,224))
                st.image(cropped_img.resize((224,224)))
            st.subheader(f'Similarity score: {similarity_score[0]*100:.2f}%')
    elif submenu3 == 'AI Driver Matching':
        pass
    elif submenu3 =='Where to go?':
        pass
    
