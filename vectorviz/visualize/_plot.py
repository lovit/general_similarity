from plotly.offline import plot, iplot
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def ipython_3d_scatter(X, color, text=None, 
        width=600, height=600, marker_size=3, colorscale='Jet'):

    data = go.Scatter3d(
        x=X[:,0],
        y=X[:,1],
        z=X[:,2],
        text = text if text else ['point #{}'.format(i) for i in range(X.shape[0])],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color,
            colorscale=colorscale,
            line=dict(
                #color='rgba(217, 217, 217, 0.14)',
                #color='rgb(217, 217, 217)',
                width=0.0
            ),
            opacity=0.8
        )
    )

    layout = go.Layout(
        autosize=False,
        width=width,
        height=height,
        margin=go.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        xaxis = dict(
            domain = [0, 1]
        )
        yaxis = dict(
            domain = [0, 1]
        )
        #paper_bgcolor='#7f7f7f',
        #plot_bgcolor='#c7c7c7'
    )

    fig = go.Figure(data=[data], layout=layout)
    iplot(fig)
