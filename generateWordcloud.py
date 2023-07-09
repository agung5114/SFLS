import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
stop_sastrawi = stop_factory.get_stop_words()

stop_custom = ['indonesia','dgn','nya','yg','jg','tuh','gue','loh',
            'sih','kok','dong','nih','aja','ga','gak','utk','and','udah','kalo','klo','kl'
            'banget','pake','banget','2022','gitu']

stop_ct = ['kabupaten','kota','provinsi','pemerintah','daerah','tanggal','tersangka','periode',
        'perkara','sprindikdik','sprin','dik','sd','utara','nama','terkait','tahun','atas']

sw = stop_sastrawi+stop_custom+ stop_ct

def create_wordcloud(df,column):
    text = df[column]
    wordcloud = WordCloud (
            background_color = 'white',
            width = 860,
            stopwords =sw,
            height = 350
                ).generate(' '.join(text))
    fig = px.imshow(wordcloud,title=f'Kata kunci terbanyak')
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})
    return fig