from dash import Dash, dcc, html, Input, Output
import dash_auth

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
    html.H1('Welcome to the app'),
    html.H3('You are successfully authorized'),
    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.H2("\n \n"),
    html.Button('Show Results', id='submit-val'),

    dcc.Graph(id='graph')
], className='container')

@app.callback(
    Output('graph', 'figure'),
    [Input('input-on-submit', 'value')])
def update_graph(value):
    
    return {
        'layout': {
            'title': 'Graph of {}',
            'margin': {
                'l': 20,
                'b': 20,
                'r': 10,
                't': 60
            }
        },
        'data': [{'x': [1, 2, 3], 'y': [4, 1, 2]}]
    }

if __name__ == '__main__':
    app.run_server(debug=True, port = 8052)