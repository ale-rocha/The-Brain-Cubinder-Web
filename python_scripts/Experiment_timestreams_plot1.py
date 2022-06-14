# Instituto Nacional de Astrofisica Optica y Electronica Mx.
# Dep. de Ciencias de la computacion
# Alejandra Rocha Solache


#Log: 9-feb-2022 @arocha : File creation
#Log: 2022-06-13 20:35:48 @arocha : File creation

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
username = "rochasolache"
api_key ='abtA3JrpQUvWsIGfaiAk'

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
import chart_studio.plotly as py
import chart_studio.tools as tools

path_experiment ="/Users/alerocha/Documents/Causal-Manifold/Experiments/Experiment1/Outputs/"

# Plot styles ------------->
#["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
theme= "plotly_white"


# Plot de los canales simulados
channels = np.genfromtxt(path_experiment+"TimeStreams/timestreams.csv", delimiter=',')
timestamps = np.genfromtxt(path_experiment+"TimeStreams/timestamp.csv", delimiter=',')
print(timestamps.shape)
channels = pd.DataFrame(data={'TS':[timestamps[:],timestamps[:],timestamps[:],timestamps[:],timestamps[:],timestamps[:],timestamps[:],timestamps[:]],
                              'Name': ['Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8'],
                              'Signal':[channels[:,0],channels[:,1],channels[:,2],channels[:,3],channels[:,4],channels[:,5],channels[:,6],channels[:,7]]})
# Plot channels
plotchannels = px.line(channels,x='TS',y='Signal',color="Name")
py.plot(plotchannels, filename="channels", auto_open=True)

