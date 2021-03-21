import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ImagesDataset
from day_night_detector import DayNightDetector
from get_data import get_all_with_field
from test import get_wrong_classified_images
from train import get_num_correct

scene_detection_col = 'scene_detection'
learning_rates = [0.001, 0.0001]
batch_sizes = [1, 10, 20]
num_workers = 4
total_correct = 0

val_data = get_all_with_field(scene_detection_col, field_name='train_test', field_value='val')
val_ds = ImagesDataset(val_data)
detector = DayNightDetector()

if __name__ == '__main__':
    print('starting validation')
    for lr in learning_rates:
        print('lr=', lr)
        for batch_size in batch_sizes:
            print('batch_size=', batch_size)
            total_correct = 0
            detector.load_state_dict(torch.load(f'models/model_lr:0.0001_bs:{batch_size}.pth'))
            detector = detector.to(device='cuda')
            detector.eval()

            tb = SummaryWriter(comment=f' validation, batch_size={batch_size}, lr={lr}')

            total = 0
            wrong = 0
            wrong_examples = []
            val_loader = DataLoader(val_ds, batch_size=20, shuffle=True, num_workers=num_workers, pin_memory=True)

            for batch in val_loader:
                images, labels, _ = batch
                total += len(images)
                labels = labels.to(device='cuda')
                images = images.to(device='cuda')

                predictions = detector(images)

                wc = get_wrong_classified_images(predictions, labels)

                wrong += len(wc)
                wrong_examples.extend(wc)

                total_correct += get_num_correct(predictions, labels)

            tb.add_scalar('total correct', total_correct)
            tb.add_scalar('Accuracy', total_correct / len(val_ds))
            tb.close()
