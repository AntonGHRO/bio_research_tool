# backend/app/main.py
import copy
import io
import traceback

import pandas as pd
import polars as pl
import torch
from fastapi import FastAPI, UploadFile, File, Request, Form, Query, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx

from backend.app.graph import get_subgraph, compute_bar_data, get_subgraph_by_label, build_graph, merge_edge_labels, \
    LinkPredModel, train, predict_new_edges


class Message(BaseModel):
    message: str

app = FastAPI()


@app.get("/hello", response_model=Message)
async def hello():
    return {"message": "Hello from Python!"}

# http://127.0.0.1:8000/subgraph?label=A
@app.post("/subgraph")
def subgraph(file: UploadFile = File(...),label: str = Query(..., description="Node label (Gene or Disease) to get subgraph for")):
    df = pd.read_csv(file.file)
    G= build_graph(df)
    return get_subgraph_by_label(G, label)

@app.post("/upload_csv_with_keys")
async def upload_csv_with_keys(
    file: UploadFile = File(...),
    key: str = Form(None),
    value: str = Form(None),
):
    """
    Receives:
      - file: CSV bytes
      - key:  name of the column to group by (e.g. 'GeneSymbol')
      - value: name of the column whose unique values to count (e.g. 'DiseaseName')

    Returns JSON:
      {
        "bar_data": { key1: count1, key2: count2, … },
        "rows": <total_rows>,
        "columns": [ … ]
      }
    """
    # 1) Read the CSV into a Polars DataFrame with relaxed schema inference
    try:
        file.file.seek(0)
        df_pl = pl.read_csv(
            file.file,
            infer_schema_length=None,
            ignore_errors=True
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR: Polars failed to parse CSV:\n{tb}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # 2) Convert to pandas for compute_bar_data
    try:
        df_pd = df_pl.to_pandas()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR: Conversion to pandas failed:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Failed to convert to pandas: {e}")

    # 3) Ensure key/value were provided
    if not key or not value:
        raise HTTPException(status_code=400, detail="Both 'key' and 'value' form fields must be supplied")

    # 4) Compute the bar chart dictionary
    try:
        bar_dict = compute_bar_data(df_pd, key, value)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR: compute_bar_data raised:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Error computing bar data: {e}")

    # 5) Return JSON with bar_data plus some metadata
    return JSONResponse(content={
        "bar_data": bar_dict,
        "rows": df_pl.height,
        "columns": df_pl.columns
    })


# @app.post("/upload_csv_with_keys")
# async def get_bar_data(
#         file: UploadFile = File(...),
#         key: str = Form(...),
#         value: str = Form(...)
# ):
#     # Read CSV content from uploaded file
#     # df = pd.read_csv(file.file)
#     #
#     # # Use your compute_bar_data function here
#     # data = compute_bar_data(df, key, value)
#
#     # response = {
#     #     "labels": list(data.keys()),
#     #     "values": list(data.values())
#     # }
#     print()
#     response = {"labels": ["a", "b", "c"], "values": [1, 2, 3]}
#     return response

@app.post("/predict-links")
async def predict_links(
    file: UploadFile = File(...),
    threshold: float = Query(0.9, ge=0.0, le=1.0)
):
    try:
        # Load dataset CSV uploaded by user
        df = pd.read_csv(file.file)

        # Build NetworkX graph from df (implement this yourself)
        G = build_graph(df)

        # Convert graph to torch_geometric Data object
        data_G = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['evidence'])
        data = data_G.clone()

        # Split edges for training/validation/testing
        transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True, split_labels=True)
        train_data, val_data, test_data = transform(data)

        train_data = merge_edge_labels(train_data)
        val_data = merge_edge_labels(val_data)
        test_data = merge_edge_labels(test_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

        # Initialize model and optimizer
        model = LinkPredModel(input_dim=train_data.num_node_features, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Train model
        best_model = None
        best_loss = float('inf')
        epochs = 50
        for epoch in range(epochs):
            loss_train, acc_train, loss_val, acc_val = train(model, train_data, val_data, optimizer)
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = copy.deepcopy(model)

        # Predict new edges above threshold on test set
        new_edges = predict_new_edges(best_model, test_data, threshold)

        return {"new_edges": new_edges, "threshold": threshold}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


