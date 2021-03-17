import numpy as np
import torch
import pymongo
from pathlib import Path


def read_movie(source_path: str, destination):
    import cv2

    cap = cv2.VideoCapture(source_path)

    print(f'splitting {source_path}')
    # Read until video is completed
    frame_num = 0

    while True:
        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret is False:
            cap.release()
            break

        movie_name = str(Path(source_path).with_suffix('').name)
        write_path = f'{destination}/{movie_name}_frame_%04d.jpg' % frame_num
        cv2.imwrite(write_path, frame)
        print(f'wrote frame {frame_num} to path {write_path}')
        frame_num += 60


def get_addresses_list(file_path):
    with open(file_path, '+r') as f:
        content = f.read()
    return content.split('\n')


def insert_list(collection_name: str, items_list: list):  # Collections: classifier , segmentation , LDS
    col = get_collection(collection_name)
    q = col.insert_many(items_list)
    print(f'inserted {len(q.inserted_ids)} to {collection_name}')
    return q


def get_collection(collection):
    client = pymongo.MongoClient(host="mongodb://guyd:GdadoNN@10.0.0.63/admin")
    db = client['local']
    col = db[collection]
    return col


def upload_mongo():
    import glob
    day_source = '/home/student/Desktop/day_addresses.txt'
    night_source = '/home/student/Desktop/night_addresses.txt'
    day_list = get_addresses_list(day_source)
    night_list = get_addresses_list(night_source)
    # path_to_data = '/mnt/data/Algo/day_night_detection'
    # day_list = glob.glob(f'{path_to_data}/**/day/**.jpg', recursive=True)
    # night_list = glob.glob(f'{path_to_data}/**/night/**.jpg', recursive=True)

    day_items = []
    for path in day_list:
        item = {
            'path_image': path,
            'optics': 'vis',
            'scene': ['day'],
            'train_test': 'val'
        }
        day_items.append(item)

    night_items = []
    for path in night_list:
        item = {
            'path_image': path,
            'optics': 'vis',
            'scene': ['night'],
            'train_test': 'val'
        }
        night_items.append(item)
    day_items.extend(night_items)

    insert_list('scene_detection', day_items)


def update_many(collection_name: str, identifier_key: str, identifier_val, field_name: str, field_value):
    collection = get_collection(collection_name)
    d = collection.update_many({identifier_key: identifier_val}, {'$set': {field_name: field_value}})
    print(f'updated {d.modified_count} items')


def get_collection_content(collection_name):
    collection = get_collection(collection_name)
    cursor = collection.find({})
    return [image for image in cursor]


def get_all_with_label(collection_name: str, label: str):
    col = get_collection(collection_name)
    r = col.find({'scene': {'$elemMatch': {'$eq': label}}})
    return list(r)


def get_all_with_field(collection_name: str, field_name: str, field_value):
    collection = get_collection(collection_name)
    cursor = collection.find({field_name: {'$eq': field_value}})
    return list(cursor)


if __name__ == '__main__':
    upload_mongo()
