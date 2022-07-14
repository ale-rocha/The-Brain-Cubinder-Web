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
    temp_phase = []
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

    #Compute curves
    temp_curve_phase_time_x = []
    temp_curve_phase_time_y = []
    temp_curve_phase_time_z = []
    temp_curve_phase_freq_x = []
    temp_curve_phase_freq_y = []
    temp_curve_phase_freq_z = []
    for i in range(0,len(temp_time)-1,1):
        ctx,cty, ctz = draw_curve (temp_phase[i],temp_phase[i+1],temp_time[i],temp_time[i+1])
        cfx,cfy,cfz = draw_curve (temp_phase[i],temp_phase[i+1],temp_freq[i],temp_freq[i+1])
        temp_curve_phase_time_x.append(ctx)
        temp_curve_phase_time_y.append(cty)
        temp_curve_phase_time_z.append(ctz)
        temp_curve_phase_freq_x.append(cfx)
        temp_curve_phase_freq_y.append(cfy)
        temp_curve_phase_freq_z.append(cfz)

    curves_channels_phase_time.append([temp_curve_phase_time_x,temp_curve_phase_time_y,temp_curve_phase_time_z])
    curves_channels_phase_freq.append([temp_curve_phase_freq_x,temp_curve_phase_freq_y,temp_curve_phase_freq_z])


print("Curves computed ", len(curves_channels_phase_time))
for curve in curves_channels_phase_time:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Creating plot
    ax.plot(np.array(curve[0]).flatten(), np.array(curve[1]).flatten(), np.array(curve[2]).flatten()) # the full spiral
    plt.show()



r1 = 1
a1 = 0
h1 = 1
x1, y1, z1 = cylinder(r1, h1, a=a1)
colorscale = [[0, 'blue'],
             [1, 'blue']]
cyl1 = go.Surface(x=x1, y=y1, z=z1,
                 colorscale = colorscale,
                 showscale=False,
                 opacity=0.3)
xb_low, yb_low, zb_low = boundary_circle(r1, h=a1)
xb_up, yb_up, zb_up = boundary_circle(r1, h=a1+h1)
bcircles1 =go.Scatter3d(x = xb_low.tolist()+[None]+xb_up.tolist(),
                        y = yb_low.tolist()+[None]+yb_up.tolist(),
                        z = zb_low.tolist()+[None]+zb_up.tolist(),
                        mode ='lines',
                        line = dict(color='blue', width=2),
                        opacity =0.55, showlegend=False)
dataFigs = [cyl1, bcircles1]
print("Curves computed ", len(curves_channels_phase_time))
for curve,name in zip(curves_channels_phase_freq,namesTraces):
    curveFig =go.Scatter3d(x=np.array(curve[0]).flatten(),
                        y=np.array(curve[1]).flatten(),
                        z=np.array(curve[2]).flatten(),
                        line = dict(width=4),
                        name = name,
                        mode='lines') # the full spiral
    dataFigs.append(curveFig)


layout = go.Layout(scene_xaxis_visible=True, scene_yaxis_visible=True, scene_zaxis_visible=True)
fig = go.Figure(data=dataFigs, layout=layout)
fig.update_layout(scene_camera_eye_z= 0.55)
fig.layout.scene.camera.projection.type = "orthographic" #commenting this line you get a fig with perspective proj
fig.update_layout(scene = dict(
                    xaxis = dict(
                        backgroundcolor="rgb(200, 200, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(200, 200, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                )
fig.update_layout(scene = dict(
                    xaxis_title=r'Cos(phase)',
                    yaxis_title=r'Sin(phase)',
                    zaxis_title=r'Frequency'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
fig.show()