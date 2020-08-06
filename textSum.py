import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime

import pandas as pd
import numpy as np

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
from PIL import Image
import pytesseract
import pdfplumber

from t5_ import textSum, preprocess
from ocr_ import ocr

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

colors = {
    "background": "#006DA2",
    "text": "#FFFFFF",
    "backgroundTab": "#006DA2",
    "textTab": "#FFFFFF",
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Home",
                    style={
                        "color": colors["textTab"],
                        "backgroundColor": colors["backgroundTab"],
                    },
                    children=[
                        html.Br(),
                        html.H1(
                            "Text Summarization Homepage",
                            style={
                                "textAlign": "center",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H5(
                            children="A tool for summarizing text documents",
                            style={
                                "textAlign": "center",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="Text to Summary - Allows input of a continuous string",
                            style={
                                "textAlign": "left",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="Summarizes string, returns output",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            children="Image to Summary - Allows input of a single image",
                            style={
                                "textAlign": "left",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="OCR to get text from string, summarizes string, returns output",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            children="PDF to Summary- Allows input of an unlimited length PDF",
                            style={
                                "textAlign": "left",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="PDFplumber gets text from each page, each page is summarized, page summaries are concatenates, splits by length 512 for longer documents, summarizes list indicies, returns output",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            children="Next Steps",
                            style={
                                "textAlign": "left",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="Visualization of model training",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.H6(
                            children="Fixed form document analysis -- Can be used for anomaly detection, RPA tasks, etc.",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.H6(
                            children="Vertical text OCR",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.H6(
                            children="Allowing handwritten text documents",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.H6(
                            children="AI modeling, text completion",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                        html.Br(),
                        html.H6(
                            children="Known Limitations",
                            style={
                                "textAlign": "left",
                                "color": colors["text"],
                                "backgroundColor": "#2D3D7B",
                            },
                        ),
                        html.H6(
                            children="T5-small being used at the moment, bert-uncased and larger BertSum models are too hardware intensive.",
                            style={
                                "textAlign": "left",
                                "color": '#2D3D7B',
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Text Summarization",
                    style={
                        "color": colors["textTab"],
                        "backgroundColor": colors["backgroundTab"],
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Br(),
                                html.H1(
                                    "Text To Summary",
                                    style={
                                        "textAlign": "center",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.H4(
                                    "Input",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                dcc.Input(
                                    id="input-1-submit",
                                    type="text",
                                    placeholder="Enter Text",
                                    maxLength=10000,
                                    size="307",
                                    persistence=True,
                                ),
                                html.Br(),
                                html.Button(
                                    "Submit",
                                    style={
                                        "textAlign": "center",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                    id="btn-submit",
                                ),
                                html.H4(
                                    "Output",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.Div(
                                    id="output-submit",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.Br(),
                                html.H1(
                                    "Image to Summary",
                                    style={
                                        "textAlign": "center",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                dcc.Upload(
                                    id="upload-image",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select File")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=True,
                                ),
                                html.H4(
                                    "Output",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.Div(
                                    id="output-image-upload",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.Br(),
                                html.H1(
                                    "PDF to Summary",
                                    style={
                                        "textAlign": "center",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                dcc.Upload(
                                    id="upload-image2",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select File")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=True,
                                ),
                                html.H4(
                                    "Output",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                                html.Div(
                                    id="output-image-upload2",
                                    style={
                                        "textAlign": "left",
                                        "color": colors["text"],
                                        "backgroundColor": "#2D3D7B",
                                    },
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="AI/Analytics Playground",
                    style={
                        "color": colors["textTab"],
                        "backgroundColor": colors["backgroundTab"],
                    },
                    children=[
                        html.Div(
                            children=[
                                
                            ]
                        ),
                    ],
                ),
            ]
        )
    ]
)

@app.callback(
    Output("output-submit", "children"),
    [Input("btn-submit", "n_clicks")],
    [State("input-1-submit", "value"),],
)

def update_output(clicked, input1):
    if clicked:
        n = 1000
        clean_x = preprocess(input1)
        if len(clean_x) > 1200:
            summ = [clean_x[i:i+n] for i in range(0, len(clean_x), n)]
            listSumm = ''
            for i in summ:
                listSumm = listSumm + textSum(i)
            print("THIS IS LISTSUM --", listSumm)
            finalSum = textSum(listSumm)
        elif len(clean_x) <= 1200:
            finalSum = textSum(clean_x)
        return finalSum

@app.callback(
    Output("output-image-upload", "children"),
    [Input("upload-image", "contents")],
    [State("upload-image", "filename"), State("upload-image", "last_modified")],
)

# Screeching. It won't return with linebreaks
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not   None:
        x = ocr(list_of_names[0])
        x_clean = preprocess(x)
        n = 1000
        summ = [x_clean[i:i+n] for i in range(0, len(x_clean), n)]
        listSumm = ''
        for i in summ:
            listSumm = listSumm + textSum(i)
        finalSum = textSum(listSumm)
        return finalSum


@app.callback(
    Output("output-image-upload2", "children"),
    [Input("upload-image2", "contents")],
    [State("upload-image2", "filename"), State("upload-image2", "last_modified")],
)
def update_output2(list_of_contents2, list_of_names2, list_of_dates2):
    if list_of_contents2 is not None:
        textSumm = ''
        finalSum = ''
        with pdfplumber.open(list_of_names2[0]) as pdf:
            pages = pdf.pages
            for i, pg in enumerate(pages):
                textList = pages[i].extract_text()
                x_clean = preprocess(textList)
                textSumm = textSumm + textSum(x_clean)
            if len(textSumm) > 1500:
                n = 1000
                summ = [x_clean[i:i+n] for i in range(0, len(x_clean), n)]
                finalSum = summ
            elif len(textSumm) <= 1500:
                finalSum = textSum(textSumm)
            return finalSum

        return "Final Summarization -- " + finalSum


if __name__ == "__main__":
    app.run_server(debug=True)
