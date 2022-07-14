import plotly.graph_objects as go
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
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
from pyrsistent import v

SMALL_SIZE = 5
MEDIUM_SIZE = 7
BIGGER_SIZE = 10

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

def minmax_norm(df,maxv):
    print("Normalizando: ")
    print("MAX value: ", max(df))
    print("MIN value: ", min(df))
    print("---------------------")
    return (df - min(df)) / ( max(df) - min(df))*maxv

def minmax_norm_serie(df,minv,maxv):
    return (df - minv) / ( maxv - minv)

path_experiment ="/Users/alerocha/Documents/Causal-Manifold/Experiments/Experiment1/Outputs/"

# Paso 1: Cargar las observaciones
eventsManifold = pd.read_csv(path_experiment+"Manifold/eventsManifold.csv")
eventsManifold.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured","PathCones"]
eventsManifold["FutureCone"] = np.nan; eventsManifold["FutureCone"] = eventsManifold["FutureCone"].astype(object)
eventsManifold["PastCone"] = np.nan;  eventsManifold["PastCone"] = eventsManifold["PastCone"].astype(object)
eventsManifold["HorismosCone"] = np.nan;  eventsManifold["HorismosCone"] = eventsManifold["HorismosCone"].astype(object)
eventsManifold["SpaceLikeCone"] = np.nan;  eventsManifold["SpaceLikeCone"] = eventsManifold["SpaceLikeCone"].astype(object)

eventsManifold["idEventsFutureCone"] = np.nan; eventsManifold["idEventsFutureCone"] = eventsManifold["idEventsFutureCone"].astype(object)
eventsManifold["idEventsPastCone"] = np.nan;  eventsManifold["idEventsPastCone"] = eventsManifold["idEventsPastCone"].astype(object)
eventsManifold["idEventsHorismosCone"] = np.nan;  eventsManifold["idEventsHorismosCone"] = eventsManifold["idEventsHorismosCone"].astype(object)
eventsManifold["idEventsSpaceLikeCone"] = np.nan;  eventsManifold["idEventsSpaceLikeCone"] = eventsManifold["idEventsSpaceLikeCone"].astype(object)

channels = np.genfromtxt(path_experiment+"TimeStreams/timestreams.csv", delimiter=',')


timestamps = np.genfromtxt(path_experiment+"TimeStreams/timestamp.csv", delimiter=',')

#Si se desea normalizar
# Caso 1: La frecuencia esta entre valores de 0 y 50
#         El tiempo es normalizado entre valores de 0 y 50
#         La fase es normalizada entre 0 y 1

eventsManifold["Frequency"] = minmax_norm(eventsManifold["Frequency"],1)
eventsManifold["Time"] = minmax_norm(eventsManifold["Time"],1)
eventsManifold["Phase"] = minmax_norm(eventsManifold["Phase"],1)
timestamps =  minmax_norm(timestamps,1)
deltaband = abs(timestamps[0] - timestamps[100])
print("Deltaband: ", deltaband)


global_min_distance_future = 1000000000
global_max_distance_future = -1000000000

#iterar sobre para obtener una matriz de distancias
for index1, row1 in eventsManifold.iterrows():
        phase1 = row1['Phase']
        frequency1 = row1['Frequency']
        time1 = row1['Time']
        distances_future= []; idevent_distances_future = []
        distances_past = []; idevent_distances_past = []
        distances_horismos = []; idevent_horismos = []
        distances_spacelike = []; idevent_spacelike = []
        for index2, row2 in eventsManifold.iterrows():
            phase2 = row2['Phase']
            frequency2 = row2['Frequency']
            time2 = row2['Time']
            distance = (phase2-phase1) + (frequency2-frequency1) - (time2-time1)
            #print("Time1: "+str(time1)+"  Time2: "+str(time2))

            if distance < 0 and time1<time2:
                distances_future.append(distance)
                idevent_distances_future.append(index2)
                if distance< global_min_distance_future:
                    global_min_distance_future = distance
                if distance> global_max_distance_future:
                    global_max_distance_future = distance

            elif distance < 0 and  time1>=time2:
                #print("Distance 2 :", distance)
                distances_past.append(distance)
                idevent_distances_past.append(index2)
            elif distance == 0:
                #print("Distance 3 :", distance)
                distances_horismos.append(distance)
                idevent_horismos.append(index2)
            elif distance > 0:
                #print("Distance 4 :", distance)
                distances_spacelike.append(distance)
                idevent_spacelike.append(index2)
            
            
            
        eventsManifold.at[index1, 'FutureCone'] = distances_future
        eventsManifold.at[index1, 'PastCone'] = distances_past
        eventsManifold.at[index1, 'HorismosCone'] = distances_horismos
        eventsManifold.at[index1, 'SpaceLikeCone'] = distances_spacelike
        eventsManifold.at[index1, 'idEventsFutureCone'] = idevent_distances_future
        eventsManifold.at[index1, 'idEventsPastCone'] = idevent_distances_past
        eventsManifold.at[index1, 'idEventsHorismosCone'] = idevent_horismos
        eventsManifold.at[index1, 'idEventsSpaceLikeCone'] = idevent_spacelike





# Graficando ====================================================================================
# ===============================================================================================
# ===============================================================================================




channels = pd.DataFrame(data={  'Hb-Ch1':channels[:,0],
                                'Hb-Ch2':channels[:,1],
                                'Hb-Ch3':channels[:,2],
                                'Hb-Ch4':channels[:,3],
                                'Hb-Ch5':channels[:,4],
                                'Hb-Ch6':channels[:,5],
                                
                                'HbO2-Ch1':channels[:,6],
                                'HbO2-Ch2':channels[:,7],
                                'HbO2-Ch3':channels[:,8],
                                'HbO2-Ch4':channels[:,9],
                                'HbO2-Ch5':channels[:,10],
                                'HbO2-Ch6':channels[:,11],

                                'Hb-Ch7':channels[:,12],
                                'HbO2-Ch7':channels[:,13],
                                'Hb-Ch8':channels[:,14],
                                'HbO2-Ch8':channels[:,15]})


for index1, row1 in eventsManifold.iterrows():

    #punto de referencia
    phase1 = row1['Phase']
    frequency1 = row1['Frequency']
    time1 = row1['Time']
    channel1 = row1['InfoChanel']
    idRelatedPoint = row1['idEventsFutureCone']  # idEventsPastCone  idEventsFutureCone
    distancesRelated = row1['FutureCone']
    infoser = row1['InfoMeasured']
    if channel1 >6 and channel1<=12:
        channel1 = channel1-6
    if channel1>12:
        channel1 = math.ceil(channel1/2)

    nameChannel1 = str(infoser) + "-Ch" + str(int(channel1))
    serie1 = channels[nameChannel1]
 
    
    #ax.axvspan(time1-1, time1, ymin=min(serie1), ymax=max(serie1), alpha=0.5, color='red')

    #Creamos una sublista del dataset que solo contenga los eventos del futuro al punto
    events_related = eventsManifold.iloc[idRelatedPoint]
    #obtenemos una lista de los canales que estan en el cono del punto
    unique_related_channels = np.unique(events_related['InfoChanel'])
    fig, axs = plt.subplots(len(unique_related_channels)+1,sharex=True,figsize=(13,7))
    axs[0].plot(timestamps,serie1, color="black")
    axs[0].legend( " p:"+str(phase1)+" f:"+str(frequency1)+" t:"+str(time1),loc="best")
    print("----------------------------------")
    print( " p:"+str(phase1)+" f:"+str(frequency1)+" t:"+str(time1))
    print("Del canal: ", nameChannel1)
    axs[0].axvspan(time1, time1+deltaband, ymin=-2, ymax=2, alpha=0.8, color='red')
    chanelcount =1
    for uc in unique_related_channels: #iteramos en cada canal
        
        print("--> Canal relacionado : ", uc)
        #Seleccionamos todos los evnetos que viene del canal uc
        events_channels_related = events_related[events_related["InfoChanel"] == uc]
        
        if len(events_channels_related) > 0:
            #Obtenemos informacion medida del canal
            measuredRelated = np.unique(events_channels_related['InfoMeasured'])
            measuredRelated = measuredRelated[0]

            #Arreglamos nombre del canal
            if uc>6 and uc<=12:
                uc = uc-6
               
            if uc>12:
                uc = math.ceil(uc/2)
             
            print("Channel number: ", uc)
            print("Meassure related: ", measuredRelated)
            nameChannelRelated = str(measuredRelated) + "-Ch" + str(int(uc))
  

            #Time related
            timesRelated = events_channels_related['Time']
            freq2 = events_channels_related['Frequency']
            time2 = events_channels_related['Time']
            phase2 = events_channels_related['Phase']
            
            for trel,f,t,p in zip(timesRelated,freq2,time2,phase2):
                d = (phase1-p) + (frequency1-f) - (time1-t)
                d = d + global_min_distance_future
                d = abs(1-minmax_norm_serie(d,global_min_distance_future,global_max_distance_future))
                #axs[chanelcount].axvspan(trel, trel+deltaband, ymin=-2, ymax=2, alpha=d, color='mediumorchid')
                #axs[chanelcount].plot(trel+(deltaband/2),p,'.', color='blue')
                #axs[chanelcount].plot(trel+(deltaband/2),f,'.', color='red')

            #Para cada evento en el canal ploteamos
            serieRelated = channels[nameChannelRelated]
            axs[chanelcount].plot(timestamps,serieRelated*3,color="black")
            #axs[chanelcount].legend(nameChannelRelated,loc="best")
            #axs[chanelcount].title.set_text(nameChannelRelated)

        chanelcount = chanelcount +1

    plt.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.show()
    

