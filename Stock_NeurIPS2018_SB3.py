import pandas as pd
from model_training import train_model
import base64
import os
from urllib.parse import quote as urlquote
from dash import Dash, dcc, html, Input, Output, State
from flask import Flask, send_from_directory
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import io
from dash import Dash, dcc, html, Input, Output
import dash_auth

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'saad': 'saad'
}
global df
df = []
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

UPLOAD_DIRECTORY = "app_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)

# app = dash.Dash(server=server)


app.layout = html.Div(
    [
        html.H2(" *********  Stock Model Trainer and Predictor *********"),

        html.H5("Select the Indicators \n"),

        dcc.Checklist(['EMA200', 'EMA89', 'MACD', "Volume"], ),

        html.H5("Input the Name of Stock you want to train model like : AAPL \n"),
        html.Div(dcc.Input(id='input-on-submit', type='text')),
        # html.Button('Submit1', id='submit-val'),

        dcc.RadioItems(id='input-radio-button',
                       options=[dict(label='Spot', value='Spot'),
                                dict(label='Futures', value='Futures')]),

        html.H5(" Or Upload the csv file of stock with stock ticker name "),
        dcc.Input(id='stock_name_value', type='text'),

        html.Div(id='container-button-basic2', children='Training not started'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.A('Upload csv/xls file')
            ]),
            style={
                "width": "20%",
                "height": "30px",
                "lineHeight": "20px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "55px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),

        html.H2("\n"),
        html.H2("\n"),

        html.Button('Start Training', id='submit-val'),
        html.Div(id='container-button-basic', children='Enter a value and press submit'),
        html.Div(id='container-button-basic1', children='Training not started'),

        html.Div(id='output-data-upload'),
        html.H5(" ---------------------------------------------------------------------------- "),

        html.H3("Training Results"),
        html.H5(id='container-button-basic3', children='No result'),
        html.H5(id='container-button-basic4', children='No result'),

        # dcc.Graph(id='graph'),

        html.Div(dcc.Input(id='resultstocks', type='text')),
        html.Button('Show results', id='result_show'),

    ], className='container'
)


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    return df


@app.callback(Output('container-button-basic', 'children'),
              Input('submit-val', 'n_clicks'),
              State('input-on-submit', 'value'),
              State('stock_name_value', 'value'))
def update_output(n_clicks, value, ticker):
    try:
        if value is not None:
            return 'Training Model on Stock {} Please wait while training completes'.format(value)
        elif ticker is not None and len(df) > 0:
            return 'Training Model on Stock {} Please wait while training completes'.format(ticker)
    except:
        pass


from glob import glob


@app.callback(Output('container-button-basic4', 'children'),
              Output('container-button-basic3', 'children'),
              Input('result_show', 'n_clicks'),
              State('resultstocks', 'value'))
def update_output(n_clicks, value):
    global gain
    try:
        if len(value) != 0:
            all_csv_files = [file
                             for path, subdir, files in os.walk("results/")
                             for file in glob(os.path.join(path, "*.csv"))]
            for files in all_csv_files:
                if files.split("/")[1].upper().find(value) != -1:
                    df_res = pd.read_csv(files)

                    x = df_res.loc[0]['total_profit']
                    y = df_res.loc[0]['win_ratio']

                    return "Winning ratio: {} ".format(y), "Gain Score achieved by model is:  {}%:".format(x)
    except:
        return "", ""


@app.callback(Output('container-button-basic2', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        parse_contents(list_of_contents, list_of_names)

        return "File Uploaded Successfully"


@app.callback(Output('container-button-basic1', 'children'),
              Input('submit-val', 'n_clicks'),
              State('input-on-submit', 'value'),
              State('stock_name_value', 'value'))
def start_training(n_clicks, value, ticker):
    if value is not None and len(df) == 0:

        train_model(value, df)
        return 'Training Completed Press Show results to see model performance'

    elif ticker is not None and type(df) is not int:
        df['tic'] = ticker
        train_model(ticker, df)

        return 'Training Completed Press Show results to see model performance'


if __name__ == "__main__":
    app.run_server(debug=True, port=8051, host='0.0.0.0')
