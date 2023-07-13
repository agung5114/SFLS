from calendar import month
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import pickle
import networkx as nx
from networkx.drawing.layout import spring_layout,random_layout

import pandas as pd
from datetime import date, timedelta
import plotly.io as pio
pio.templates.default = "plotly_dark"

import scipy.stats as stats
from statistics import stdev
import math

def create_network(df,Source,Target,nodecol,edgecol,val):
    label1 = df.groupby(by=[Source],as_index=False).agg({val:'sum'})
    label2 = df.groupby(by=[Target],as_index=False).agg({val:'sum'})
    label1.columns = ['entity','value']
    label2.columns = ['entity','value']
    labels = pd.concat([label1,label2])
    nodeval = labels.set_index('entity').to_dict()['value']
    G = nx.from_pandas_edgelist(df,Source,Target,create_using=nx.Graph)
    P = random_layout(G)
    nx.set_node_attributes(G, P, 'pos')
    deg = dict(G.degree)
    deglist = [v for v in deg.values()]
    node_range = stats.zscore(np.log(np.array(deglist)))
    # node_size= [v +10 if v<100 else 100+(v*10/100) for v in deg.values()]
    node_size= [v +15 if v<1 else 15+(2**v) for v in node_range.tolist()]
    node_size = [0 if math.isnan(x) else x for x in node_size]
    edge_x = []
    edge_y = []
    for edge in G.edges():
        # print(G.nodes[edge[0]]['pos'])
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        hoverinfo='none',
        mode='lines',
        line=dict(width=0.5, color=edgecol))

    #nodes
    node_x = []
    node_y = []
    node_z = []
    z_text = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        # node_z.append([node,G.degree(node)])
        node_x.append(x)
        node_y.append(y)
        node_z.append(f'{node}: {G.degree(node)} network({"{:,.2f}".format(nodeval[node]/1000000)}millions)')
        # z_text.append(val[node])
        if G.degree(node)>=0.5*stdev(deglist):
            z_text.append(node)
        else:
            z_text.append(None)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=z_text,
        mode='markers+text',
        hoverinfo='text',
        # hovertext = [G.degree(node)],
        hovertemplate=node_z,
        marker=dict(
            showscale=False,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=nodecol,
            reversescale=False,
            color= node_size,
            # color= color_p,
            # size = node_size,
            line_width=2,))

    node_trace.marker.size = node_size
    return edge_trace,node_trace

def draw_network(df,Source,Target,nodecol, edgecol,val):
    # nodecol = 'Reds'
    # edgecol = 'teal'
    net = create_network(df, Source, Target, nodecol, edgecol,val)
    net = go.Figure(data=[net[0], net[1]],
                     layout=go.Layout(
                         titlefont_size=16,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
    return net

# def save_network(df, idtopic,year,month):
#     df = df[df['Keyword']==idtopic]
#     edgeall = df
#     netall = draw_network(edgeall, 'Username', 'Retweeters', 'Burg', 'teal')

#     edge1 = df[df['Sentiment'].isin(['negative'])]
#     net1 = draw_network(edge1, 'username', 'rtusername', 'Burg', 'teal')

#     edge2 = df[df['Sentiment'].isin(['positive'])]
#     net2 = draw_network(edge2, 'username', 'rtusername', 'Blugrn', 'orange')
    
#     edge3 = df[df['Sentiment'].isin(['neutral'])]
#     net3 = draw_network(edge3, 'username', 'rtusername', 'viridis', 'sandybrown')

#     networkname = f'net_{idtopic}_{year}_{month}'
#     netall.write_html(f'all_{networkname}.html')
#     net1.write_html(f'neg_{networkname}.html')
#     net2.write_html(f'pos_{networkname}.html')
#     net3.write_html(f'neu_{networkname}.html')
#     return print(f"sukses create network bulan {month}-{year}")

# df = pd.read_csv('retweets.csv')
# # print(df.head())
# save_network(df,'kemenkeu','2023','1')