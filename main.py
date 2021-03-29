import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ImagesDataset
from day_night_detector import DayNightDetector
from get_data import get_all_with_field
from test import get_wrong_classified_images
from train import get_num_correct

scene_detection_col = 'scene_detection'
# learning_rates = [0.001, 0.0001]
# batch_sizes = [1, 10, 20]
num_workers = 0
# total_correct = 0
batch_size = 20
# lr = 0.0001
# #
val_data = get_all_with_field(scene_detection_col, field_name='train_test', field_value='val')
val_ds = ImagesDataset(val_data)
train_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#
# detector = torchvision.models.resnet50(pretrained=True)

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from efficientnet_pytorch import EfficientNet

    images, labels, _ = next(iter(train_loader))
    # grid = torchvision.utils.make_grid(images)
    tb = SummaryWriter()
    model = EfficientNet.from_pretrained('efficientnet-b0')
    print(model._fc)



