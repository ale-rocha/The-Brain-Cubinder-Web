# Instituto Nacional de Astrofisica Optica y Electronica Mx.
# Dep. de Ciencias de la computacion
# Alejandra Rocha Solache


#Log: 9-feb-2022 @arocha : File creation
#Log: 2022-06-13 20:35:48 @arocha : File creation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams as rc
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import plotly.express as px
import os
import plotly.graph_objects as go

plt.style.use('classic')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

path_experiment ="/Users/alerocha/Documents/Causal-Manifold/Experiments/Experiment1/Outputs/"

# Plot styles ------------->
#["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
theme= "plotly_white"


# Plot de los canales simulados
channels = np.genfromtxt(path_experiment+"TimeStreams/timestreams.csv", delimiter=',')
timestamps = np.genfromtxt(path_experiment+"TimeStreams/timestamp.csv", delimiter=',')
print(timestamps.shape)
channels = pd.DataFrame(data={  'Ch1-Oxy':channels[:,0],
                                'Ch2-Oxy':channels[:,1],
                                'Ch3-Oxy':channels[:,2],
                                'Ch4-Oxy':channels[:,3],
                                'Ch5-Oxy':channels[:,4],
                                'Ch6-Oxy':channels[:,5],
                                'Ch1-DeOxy':channels[:,6],
                                'Ch2-DeOxy':channels[:,7],
                                'Ch3-DeOxy':channels[:,8],
                                'Ch4-DeOxy':channels[:,9],
                                'Ch5-DeOxy':channels[:,10],
                                'Ch6-DeOxy':channels[:,11],
                                'Control1-Oxy':channels[:,12],
                                'Control1-DeOxy':channels[:,13],
                                'Control2-Oxy':channels[:,14],
                                'Control2-DeOxy':channels[:,15]})

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch1-Oxy'], color="blue", lw=4, label="OxyHb")
plt.plot(timestamps,channels['Ch1-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 1"); plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch2-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels['Ch2-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 2")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch3-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels['Ch3-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 3")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch4-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels['Ch4-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 4")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch5-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels['Ch5-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 5")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels['Ch6-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels['Ch6-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title("Channel 6")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels[ 'Control1-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels[ 'Control1-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title(" Control1-Oxy")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.figure( figsize=(6,6),facecolor='white')
plt.plot(timestamps,channels[ 'Control2-Oxy'], color="blue", lw=4,label="OxyHb")
plt.plot(timestamps,channels[ 'Control2-DeOxy'],color = "red", lw=4,label="DeoxyHb")
plt.axhline(y=0,xmin=0,xmax=timestamps.shape[0],color="black", linestyle = '--',alpha=0.4)
plt.title(" Control1-Oxy")
plt.legend(loc="best")
plt.xlabel("Time [s]"); plt.ylabel(r'$\Delta c$(OxyHb, DeoxyHb)/$\mu$M')

plt.show()

'''

# Plot deL GRID ------------------------------------------------------------------------
gridManifold = np.genfromtxt(path_experiment+"Manifold/gridManifold.csv", delimiter=',')
gridManifold = pd.DataFrame(data={'Phase':gridManifold[:,0],
                                'PhaseSin':gridManifold[:,1],
                                'PhaseCos':gridManifold[:,2],
                                'Frequency':gridManifold[:,3],
                                'Time':gridManifold[:,4],
                                'InfoChanel':gridManifold[:,5]})
# ------------------------------------------------------------------ phase - FREQUENCT
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
# Add x, y gridlines
ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.3, alpha = 0.2)
# Creating color map
my_cmap = plt.get_cmap('hsv')
# Creating plot
sctt = ax.scatter3D(gridManifold['PhaseSin'], gridManifold['PhaseCos'], gridManifold['Frequency'],
                    alpha = 0.2,
                    cmap = my_cmap,
                    marker ='^')
plt.title("PHASE - FREQUENCY")
ax.set_xlabel('COS(PHASE)', fontweight ='bold')
ax.set_ylabel('SIN(PHASE)', fontweight ='bold')
ax.set_zlabel('FREQUENCY', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

# ------------------------------------------------------------------ phase - TIME
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
# Add x, y gridlines
ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.3, alpha = 0.2)
# Creating color map
my_cmap = plt.get_cmap('hsv')
# Creating plot
sctt = ax.scatter3D(gridManifold['PhaseSin'], gridManifold['PhaseCos'], gridManifold['Time'],
                    alpha = 0.2,
                    cmap = my_cmap,
                    marker ='^')
plt.title("PHASE - TIME")
ax.set_xlabel('COS(PHASE)', fontweight ='bold')
ax.set_ylabel('SIN(PHASE)', fontweight ='bold')
ax.set_zlabel('TIME', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

# ------------------------------------------------------------------ phase - freq - time
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
# Add x, y gridlines
ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.3, alpha = 0.2)
# Creating color map
my_cmap = plt.get_cmap('hsv')
# Creating plot
sctt = ax.scatter3D(gridManifold['PhaseCos'], gridManifold['Frequency'], gridManifold['Time'],
                    alpha = 0.2,
                    cmap = my_cmap,
                    marker ='^')
plt.title("PHASE - TIME")
ax.set_xlabel('COS(PHASE)', fontweight ='bold')
ax.set_ylabel('FREQ', fontweight ='bold')
ax.set_zlabel('TIME', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.show()

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# Plot deL EVENTOS ------------------------------------------------------------------------
eventsManifold = pd.read_csv(path_experiment+"Manifold/eventsManifold.csv")
eventsManifold.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured","PathCones"]
# ------------------------------------------------------------------ phase - FREQUENCT
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
fig3 = px.scatter_3d(gridManifold, x='PhaseSin', y='PhaseCos', z='Frequency',color='Frequency', opacity = 0.1,title="phase - time")
fig = px.line_3d(eventsManifold, x='PhaseSin', y='PhaseCos', z='Frequency',color='InfoChanel',title="phase - frequency")
fig2 = px.scatter_3d(eventsManifold, x='PhaseSin', y='PhaseCos', z='Frequency',color='InfoChanel',title="phase - frequency")
fig.add_trace(fig2.data[0])
fig.add_trace(fig3.data[0])
fig.update_coloraxes(showscale=False)
fig.show()

# ------------------------------------------------------------------ phase - TIME
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
fig3 = px.scatter_3d(gridManifold, x='PhaseSin', y='PhaseCos', z='Time',color='Time', opacity = 0.1,title="phase - time")
fig = px.line_3d(eventsManifold, x='PhaseSin', y='PhaseCos', z='Time',color='InfoChanel',title="phase - time")
fig2 = px.scatter_3d(eventsManifold, x='PhaseSin', y='PhaseCos', z='Time',color='InfoChanel',title="phase - time")
fig.add_trace(fig2.data[0])
fig.add_trace(fig3.data[0])
fig.update_coloraxes(showscale=False)
fig.show()

# ------------------------------------------------------------------ phase - freq - time
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
fig3 = px.scatter_3d(gridManifold, x='PhaseSin', y='Frequency', z='Time',color='Time', opacity = 0.1,title="phase - frequency- time")
fig = px.line_3d(eventsManifold, x='PhaseSin', y='Frequency', z='Time',color='InfoChanel',title="phase - frequency- time")
fig2 = px.scatter_3d(eventsManifold, x='PhaseSin', y='Frequency', z='Time',color='InfoChanel',title="phase - frequency- time")
fig.add_trace(fig2.data[0])
fig.add_trace(fig3.data[0])
fig.update_coloraxes(showscale=False)
fig.show()

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

for index, row in eventsManifold.iterrows():
    name = row['PathCones']
    print(row)

    if os.stat(path_experiment+"Cones/"+str(name)+"/futureCone.csv").st_size != 0:
        cone = pd.read_csv(path_experiment+"Cones/"+str(name)+"/futureCone.csv")
        cone.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured"]
        fig = px.scatter_3d(cone, x='Phase', y='Frequency', z='Time',title="future")
        fig2 = px.scatter_3d(gridManifold, x='Phase', y='Frequency', z='Time',color ='Time',opacity = 0.1)
        fig.add_trace(fig2.data[0])
        fig.show()

    

    if os.stat(path_experiment+"Cones/"+str(name)+"/pastCone.csv").st_size != 0:
        cone = pd.read_csv(path_experiment+"Cones/"+str(name)+"/pastCone.csv")
        cone.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured"]
        fig =  px.scatter_3d(cone, x='Phase', y='Frequency', z='Time',title="Past cone")
        fig2 = px.scatter_3d(gridManifold, x='Phase', y='Frequency', z='Time',color ='Time',opacity = 0.1)
        fig.add_trace(fig2.data[0])
        fig.show()

    if os.stat(path_experiment+"Cones/"+str(name)+"/horismosCone.csv").st_size != 0:
        cone = pd.read_csv(path_experiment+"Cones/"+str(name)+"/horismosCone.csv")
        cone.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured"]
        fig =  px.scatter_3d(cone,x='Phase', y='Frequency', z='Time', title="horismos cone")
        fig2 = px.scatter_3d(gridManifold, x='Phase', y='Frequency', z='Time',color ='Time',opacity = 0.1)
        fig.add_trace(fig2.data[0])
        fig.show()

'''