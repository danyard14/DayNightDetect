import matplotlib
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from data_loader import ImagesDataset
from day_night_detector import DayNightDetector
from get_data import get_all_with_field
from test import model_path

writer = SummaryWriter('runs/scene_detection_images')
val_data = get_all_with_field('scene_detection', field_name='train_test', field_value='val')
val_ds = ImagesDataset(val_data)
val_loader = DataLoader(val_ds, batch_size=40, shuffle=True, num_workers=0, pin_memory=True)
dataiter = iter(val_loader)
images, labels, _ = next(dataiter)

if __name__ == '__main__':
    # create grid of images
    images.to(device='cuda')
    labels.to(device='cuda')
    detector = DayNightDetector()
    detector.load_state_dict(torch.load(model_path))

    img_grid = torchvision.utils.make_grid(images)
    # show images
    writer.add_image('images', img_grid)
    writer.add_graph(detector, images)
    writer.close()
    # detector = detector.to(device='cuda')
    # detector.eval()
    # # write to tensorboard
    # writer.add_graph(detector, images)
    # writer.close()