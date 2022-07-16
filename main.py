from ipaddress import collapse_addresses
from fastapi import FastAPI, File
from pymongo import MongoClient
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import humps
import collections


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
def root(img: bytes = File()):
    ssim_measures = {}
    src_img = get_cv_img(img)
    width = int(src_img.shape[1] * 100 / 100)
    height = int(src_img.shape[0] * 100 / 100)
    dim = (width, height)
    images = get_images_from_mongo()
    for key, value in images.items():
        ssim_measures[key] = []
        for image in value:
            data_img = get_cv_img(image)
            resized_img = cv2.resize(
                data_img, dim, interpolation=cv2.INTER_AREA)
            ssim_measures[key].append(ssim(src_img, resized_img))

    avg = {}
    for key, value in ssim_measures.items():
        avg[str(key)] = sum(ssim_measures[key]) / len(ssim_measures[key])
    sorted_sim = {k: v for k, v in sorted(avg.items(), key=lambda item: item[1], reverse=True)}
    top_10_items = get_top_10_items(sorted_sim)
    top_10_items = humps.decamelize(top_10_items)
    return top_10_items


def get_database():
    CONNECTION_STRING = "mongodb+srv://fgroup:WCPcOjQZ5IENAqz3@cluster0.lanii.mongodb.net/admin?authSource=admin&replicaSet=atlas-kmhnrd-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true"
    client = MongoClient(CONNECTION_STRING, fsync=True)
    return client['FindX']


def get_images_from_mongo():
    db = get_database()
    collection = db['items']
    items = collection.find()
    images = get_images_from_items(items)
    return images


def get_images_from_items(items):
    items_images = {}
    for item in items:
        items_images[item['_id']] = item['Images']
    return items_images


def get_top_10_items(top_10):
    ids = []
    items = []
    for key, value in top_10.items():
        ids.append(key)
    db = get_database()
    mongo_items = db['items'].find({"_id": {"$in": ids}})
    for item in mongo_items:
        items.append(item)
    for itm in items:
        imgs = []
        for img in itm['Images']:
            imgs.append(base64.b64encode(img).decode('utf-8'))
        itm['Images'] = imgs
    sorted_items = []
    for id in ids:
        for item in items:
            if item['_id'] == id:
                sorted_items.append(item)
    return sorted_items


def get_cv_img(source):
    img = np.asarray(bytearray(source), dtype=np.uint8)
    cv2_img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img
    # return cv2.imdecode(source, cv2.IMREAD_UNCHANGED)

def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Structural Simularity Index
    """
    _assert_image_shapes_equal(org_img, pred_img, "SSIM")

    return structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )
    assert org_img.shape == pred_img.shape, msg
