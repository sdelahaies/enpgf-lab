import dash
import dash_bootstrap_components as dbc
import itertools
from dash import Input, Output, dcc, html,State
import numpy as np
import plotly.graph_objects as go
import dash_cytoscape as cyto
import datetime
import base64
from dash.exceptions import PreventUpdate
import json

def load_data_fct(sim_data):
    if sim_data is not None:
        load_data = html.Div([
                dcc.Upload(
                    id='upload-sim',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                html.Div(id='data-info',
                        children=[
                html.H3(f'data: {sim_data["enpgf"]["fname"]}'),
                html.P(f'number of nodes: {sim_data["enpgf"]["n"]}'),
                html.P(f'ensemble size: {sim_data["enpgf"]["nens"]}'),
                html.P(f'number of iterations: {sim_data["enpgf"]["nstep"]}'),
                html.P(f'history file: {sim_data["enpgf"]["dN"]}'),
                html.P(f'excitation matrix file: {sim_data["enpgf"]["alpha"]}')]
                )
        ])
    else:
        load_data = html.Div([
                dcc.Upload(
                    id='upload-sim',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                html.Div(id='data-info',
                        children=[
                html.H3(f'data: ...'),
                html.P(f'number of nodes: ...'),
                html.P(f'ensemble size: ...'),
                html.P(f'number of iterations: ...'),
                html.P(f'history file: ...'),
                html.P(f'excitation matrix file: ...')]
                )
        ])
    return load_data


def load_network(sim_data):
    if sim_data is None:
        return []
    fname = sim_data["enpgf"]["alpha"]
    alpha= np.loadtxt(fname,delimiter=",",dtype=float)
    alpha=np.round(alpha,2)


    network_data = [
        f"{i} {j} {alpha[i, j]}"
        for i, j in itertools.product(range(128), range(128))
        if alpha[i, j] > 0.1
    ]

    edges = network_data
    nodes = set()

    cy_edges = []
    cy_nodes = []

    for network_edge in edges:
        source, target, strength = network_edge.split(" ")

        if source not in nodes:
            nodes.add(source)
            cy_nodes.append({"data": {"id": source, "label": source}})
        if target not in nodes:
            nodes.add(target)
            cy_nodes.append({"data": {"id": target, "label": target}})

        cy_edges.append({
            'data': {
                'source': source,
                'target': target,
                'strength':strength
            }
        })

    default_stylesheet = [
        {
            "selector": 'node',
            'style': {
                "label": "data(label)",
                "color": "#d6ccbe",
                "opacity": 0.95,       
            }
        },
        {
            "selector": 'edge',
            'style': {
                "curve-style": "bezier",
                "opacity": 0.05
            }
        },
    ]


    return html.Div(
        [
            html.Div(
                className='eight columns',
                children=[
                    cyto.Cytoscape(
                        id='cytoscape',
                        elements=cy_edges + cy_nodes,
                        layout={'name': 'concentric'},
                        style={
                            "label": "data(label)",
                            'height': '95vh',
                            'width': '100%',
                            "font-size": 12,
                        },
                        stylesheet=default_stylesheet,
                    )
                ],
            ),
        ]
    )


def load_matrix(sim_data):
    if sim_data is None:
        return []
    fname = sim_data["enpgf"]["alpha"]
    alpha= np.loadtxt(fname,delimiter=",",dtype=float)
    alpha=np.round(alpha,2)

    fig1 = go.Figure(data=go.Heatmap(
            z = np.flip(alpha,axis=0),
            type = 'heatmap',
            colorscale = 'Inferno'),
            layout= {'width':800,
                    'height':800,
                    'autosize': False,
                    'font_color':"#d6ccbe",
                    'paper_bgcolor':'rgba(0,0,0,0)',
                    'plot_bgcolor':'rgba(0,0,0,0)'
                    })
    return dcc.Graph(figure=fig1)   
 

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY],suppress_callback_exceptions=True)
app.title = "EnPGF Lab"

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    #"background-color": "#f8f9fa",
    "background-color": "#0f0d14"
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H5("EnPGF Lab"),
        html.Hr(),
        dbc.Nav(
            [   
                dbc.NavLink("Home", href="/", active="exact"),
                #dbc.NavLink("run sim", href="/run_sim", active="exact"),
                #dbc.NavLink("load sim", href="/load_sim", active="exact"),
                dbc.NavLink("network", href="/network", active="exact"),
                dbc.NavLink("matrix", href="/matrix", active="exact"),
                #dbc.NavLink("kafka", href="/kafka", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
                dcc.Store(id='sim-data'),
                dcc.Location(id="url"),
                sidebar,
                content
                ])


@app.callback(Output("page-content", "children"),
              Input("url", "pathname"),
              State('sim-data','data')
              )
def render_page_content(pathname,sim_data):
    print(sim_data)
    if pathname == "/":
        return load_data_fct(sim_data)
    elif pathname == "/network":
        return load_network(sim_data)
    elif pathname == "/matrix":
        return load_matrix(sim_data)
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('cytoscape', 'tapNode')],
              [State("url", "pathname")])
def generate_stylesheet(node,pathname):
    if pathname != "/network":
        return
    if not node:
        raise PreventUpdate
    node_shape='circle'
    stylesheet = [
        {"selector": 'node', 'style': {'opacity': 0.4, 'shape': node_shape}},
        {
            'selector': 'edge',
            'style': {
                "label": "data(strength)",
                "color": "#d6ccbe",
                "font-size": 12,
                'opacity': 0.05,
                "curve-style": "bezier",
            },
        },
        {
            "selector": f"""node[id = "{node['data']['id']}"]""",
            "style": {
                'background-color': 'green',
                "color": "#d6ccbe",
                "border-color": "yellow",
                "border-width": 2,
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "#d6ccbe",
                "text-opacity": 1,
                "font-size": 13,
                'z-index': 9999,
            },
        },
    ]

    follower_color='#0074D9'
    following_color='#FF4136'
    for edge in node['edgesData']:
        if edge['source'] == node['data']['id']:
            stylesheet.extend(
                (
                    {
                        "selector": f"""node[id = "{edge['target']}"]""",
                        "style": {
                            "label": "data(label)",
                            "color": "#d6ccbe",
                            "font-size": 12,
                            'background-color': following_color,
                            'opacity': 0.9,
                        },
                    },
                    {
                        "selector": f"""edge[id= "{edge['id']}"]""",
                        "style": {
                            "mid-target-arrow-color": following_color,
                            "mid-target-arrow-shape": "vee",
                            "line-color": following_color,
                            'opacity': 0.9,
                            'z-index': 5000,
                        },
                    },
                )
            )
        if edge['target'] == node['data']['id']:
            stylesheet.extend(
                (
                    {
                        "selector": f"""node[id = "{edge['source']}"]""",
                        "style": {
                            "label": "data(label)",
                            "color": "#d6ccbe",
                            "font-size": 12,
                            'background-color': follower_color,
                            'opacity': 0.9,
                            'z-index': 9999,
                        },
                    },
                    {
                        "selector": f"""edge[id= "{edge['id']}"]""",
                        "style": {
                            "mid-target-arrow-color": follower_color,
                            "mid-target-arrow-shape": "vee",
                            "line-color": follower_color,
                            'opacity': 1,
                            'z-index': 5000,
                        },
                    },
                )
            )
    return stylesheet


@app.callback(Output('data-info', 'children'),
              Output('sim-data','data'),
              Input('upload-sim', 'contents'),
              State('upload-sim', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        content_type,content_string =list_of_contents.split(',')
        data=base64.b64decode(content_string)
        data_dict = json.loads(data.decode('utf8'))
        children = [
            html.Hr(),
            html.H3(f"data: {list_of_names}"),
            html.P(f'number of nodes: {data_dict["enpgf"]["n"]}'),
            html.P(f'ensemble size: {data_dict["enpgf"]["nens"]}'),
            html.P(f'number of iterations: {data_dict["enpgf"]["nstep"]}'),
            html.P(f'history file: {data_dict["enpgf"]["dN"]}'),
            html.P(f'excitation matrix file: {data_dict["enpgf"]["alpha"]}'),
            ]
        return children,data_dict
    else:
        raise PreventUpdate
        

if __name__ == "__main__":
    app.run_server(host='0.0.0.0',port=8889,debug=False)