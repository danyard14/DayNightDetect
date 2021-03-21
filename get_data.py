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


def upload_mongo(files_list: list, label='day', train_test='train'):
    assert label in ['day', 'night'], f'label should be day or night but given {label}'
    assert train_test in ['train', 'val', 'test'], f'train_test should be train or val or test but given {train_test}'

    from pathlib import Path

    items = []
    for path in files_list:
        if not Path(path).exists():
            print('path does not exist', path)
            continue
        item = {
            'path_image': path,
            'optics': 'vis',
            'scene': [label],
            'train_test': train_test
        }
        items.append(item)

    insert_list('scene_detection', items)


def upload_mongo_text_files():
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


files = "/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000000_29dc5f25-76af-11eb-a7cc-000000000000.jpg \
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000001_29fecbb7-76af-11eb-90bb-000000000001.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000002_2a08646d-76af-11eb-aff1-000000000002.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000003_2a13c1d9-76af-11eb-84cb-000000000003.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000004_2a2106b8-76af-11eb-9061-000000000004.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000005_2a2d6c04-76af-11eb-a95b-000000000005.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000006_2a389f6c-76af-11eb-be9a-000000000006.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000007_2a44b6ae-76af-11eb-bc04-000000000007.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000008_2a51f55c-76af-11eb-a595-000000000008.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000009_2a5cbdc0-76af-11eb-9601-000000000009.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000010_2a6a98fc-76af-11eb-b1fa-00000000000a.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000011_2a797434-76af-11eb-b8a6-00000000000b.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000012_2a83d578-76af-11eb-8fe2-00000000000c.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000013_2a8c5f3f-76af-11eb-bd65-00000000000d.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000014_2a96646a-76af-11eb-a358-00000000000e.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000332_3544e53b-76af-11eb-a11e-00000000014c.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000333_35588364-76af-11eb-abe6-00000000014d.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000334_356315ad-76af-11eb-84b6-00000000014e.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000335_35696ecb-76af-11eb-9fd8-00000000014f.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000336_35741669-76af-11eb-a5e0-000000000150.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000337_3579d06d-76af-11eb-8096-000000000151.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000338_357fc7fd-76af-11eb-9544-000000000152.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000339_358eb7b9-76af-11eb-a2bd-000000000153.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000340_35966373-76af-11eb-a3aa-000000000154.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000341_359eec9a-76af-11eb-9a60-000000000155.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000342_35a96bfe-76af-11eb-9a1f-000000000156.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000343_35b1b078-76af-11eb-81ab-000000000157.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000344_35b8f51d-76af-11eb-bf58-000000000158.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000345_35c6d303-76af-11eb-80d9-000000000159.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000346_35d1b733-76af-11eb-9fda-00000000015a.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000347_35e2be19-76af-11eb-951d-00000000015b.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000348_35ee15dd-76af-11eb-94d9-00000000015c.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000349_35f7e021-76af-11eb-892e-00000000015d.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000350_3600d99b-76af-11eb-a80f-00000000015e.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000351_360aad2c-76af-11eb-8099-00000000015f.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000352_361a3bdd-76af-11eb-ada3-000000000160.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000353_3624696d-76af-11eb-b7c1-000000000161.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000354_362fc421-76af-11eb-ae1e-000000000162.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000355_3639127b-76af-11eb-9e47-000000000163.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000356_36414f1c-76af-11eb-a5da-000000000164.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000357_364d2412-76af-11eb-854e-000000000165.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000371_36eb847f-76af-11eb-8369-000000000173.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000372_36f9104a-76af-11eb-9c34-000000000174.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000373_3701d715-76af-11eb-a099-000000000175.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000374_370de9f5-76af-11eb-a6e6-000000000176.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000375_371c15e0-76af-11eb-9e6d-000000000177.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000376_37259281-76af-11eb-93c5-000000000178.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000377_37313128-76af-11eb-8979-000000000179.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000378_3739415e-76af-11eb-8c02-00000000017a.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000379_37400ec5-76af-11eb-a602-00000000017b.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000380_374e7a2f-76af-11eb-992d-00000000017c.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000381_3756d92c-76af-11eb-afc0-00000000017d.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000382_3760f7be-76af-11eb-bab7-00000000017e.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000383_376a130e-76af-11eb-87a4-00000000017f.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000384_3774b050-76af-11eb-82ce-000000000180.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000385_377dbe04-76af-11eb-ad74-000000000181.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000386_37893de1-76af-11eb-bdd2-000000000182.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000387_3799fdc8-76af-11eb-9e38-000000000183.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000388_37a35189-76af-11eb-b739-000000000184.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000389_37ada429-76af-11eb-ac3b-000000000185.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000390_37b8563c-76af-11eb-8ae8-000000000186.jpg\
/mnt/data/Algo/dataset/foresight/detection/hard_negative/VIS/items/000391_37cb33c9-76af-11eb-9aa8-000000000187.jpg"



if __name__ == '__main__':
    upload_mongo()
