import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
fig = plt.figure()
ax = fig.gca(projection='3d')

path_experiment ="/Users/alerocha/Documents/Causal-Manifold/Experiments/Experiment1/Outputs/"


def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z


def draw_curve(phase1,phase2,z0,z1):
    """
    Draw a spiral curve between 2 points
    """
    if phase1 == -3.1416:
        phase1 = 3.1416
    if phase2 == -3.1416:
        phase2 = 3.1416
    if abs(phase1 - phase2)>=np.pi:
        dir = 1
    else:
        dir = 1
    theta  = np.linspace(phase1, phase2, 100, endpoint=True)
    thetaz = np.linspace(z0, z1, 100, endpoint=True)
    helix_x = dir*np.cos(1*theta)
    helix_y = dir*np.sin(1*theta)
    helix_z = thetaz
    return helix_x, helix_y, helix_z


# Paso 1: Cargar las observaciones
eventsManifold = pd.read_csv(path_experiment+"Manifold/eventsManifold.csv")
eventsManifold.columns = ["Phase", "PhaseSin", "PhaseCos", "Frequency","Time","InfoChanel","InfoMeasured","PathCones"]
unique_Channels = np.unique(eventsManifold['InfoChanel'])
namesTraces = []
curves_channels_phase_time = []
curves_channels_phase_freq = []

for channel in unique_Channels:
    #subsampling
    temp_phase = []s
    temp_freq = []
    temp_time = []
    namesTraces.append("Channel: "+str(channel))
    print(" [->] Computing curve for channel :", channel)
    current_channel = eventsManifold[eventsManifold["InfoChanel"] == channel]
    current_channel = current_channel.reset_index()
    for index, row in current_channel.iterrows():
        temp_phase.append(row['Phase'])
        temp_freq.append(row['Frequency'])
        temp_time.append(row['Time'])

