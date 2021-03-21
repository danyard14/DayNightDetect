import torch
import pymongo
import day_night_detector
from day_night_detector import DayNightDetector
from data_loader import ImagesDataset
from torch.utils.data import DataLoader, Dataset
from torch import optim
from get_data import get_collection_content

model_path = 'models/model_1.pth'
learning_rate = 0.0001
total_loss = 0
total_correct = 0
# data_folder = '/content/drive/MyDrive/day_night_detection_data/test'
batch_size = 1
epochs = 2


def get_wrong_classified_images(preds, labels, images_paths=None):
    if images_paths is None:
        images_paths = [None for _ in preds]
    wrong_predictions = preds.argmax(dim=1).ne(labels)
    return [im_path for im_path, wrong in zip(images_paths, wrong_predictions) if wrong]


if __name__ == '__main__':
    import os
    data = get_collection_content('scene_detection')

    print('starting test')

    test_loader = DataLoader(ImagesDataset(data, lambda x: True if 'day' in x['scene'] else False), batch_size, shuffle=False, num_workers=2, pin_memory=True)
    detector = DayNightDetector()

    detector.load_state_dict(torch.load(model_path))
    detector = detector.to(device='cuda')
    detector.eval()

    total = 0
    wrong = 0
    wrong_examples = []

    for images, labels, images_paths in test_loader:
        total += len(images)
        labels = labels.to(device='cuda')
        images = images.to(device='cuda')

        preds = detector(images)

        wc = get_wrong_classified_images(preds, labels, images_paths)

        wrong += len(wc)
        wrong_examples.extend(wc)



# model = Model(arguments)
# model.load_state_dict(torch.load(PATH))
# model.eval() // if we want to evaluate and not train again