from torch.utils.tensorboard import SummaryWriter
from day_night_detector import DayNightDetector
from data_loader import ImagesDataset
from torch.utils.data import DataLoader
from get_data import get_all_with_field
from torch import optim
import torch
from torch.nn import functional
from pathlib import Path

scene_detection_col = 'scene_detection'
learning_rates = [0.001, 0.0001]
total_loss = 0
total_correct = 0
data_folder = '/home/student/Desktop/Dan/day_night_detection'
batch_sizes = [1, 10, 20]
epochs = 2
num_workers = 0

train_data = get_all_with_field(scene_detection_col, field_name='train_test', field_value='train')
train_ds = ImagesDataset(train_data)

mark_every = len(train_ds) // 20
mark_split = [(start, start + mark_every - 1) for start in range(0, len(train_ds), mark_every)]
mark_index = 0


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def should_mark(num_processed: int) -> bool:
    global mark_every, mark_split, mark_index

    if num_processed > mark_split[mark_index][1]:
        mark_index += 1
        return True
    return False


if __name__ == '__main__':

    print('starting training')
    for lr in learning_rates:
        print('lr=', lr)
        for batch_size in batch_sizes:
            print('batch_size=', batch_size)

            detector = DayNightDetector()
            detector = detector.to(device='cuda')
            file_name = f'models/model_lr:{lr}_bs:{batch_size}.pth'
            if Path(file_name).exists():
                detector.load_state_dict(torch.load(file_name))

            # train
            detector.train()
            optimizer = optim.Adam(detector.parameters(), lr=lr)
            train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

            if not Path(f'models/model_{lr}_{batch_size}.pth').exists():
                tb = SummaryWriter(comment=f' train, batch_size={batch_size} lr={lr}')

                for epoch in range(epochs):
                    print('\t train epoch', epoch)
                    total_correct, batch_counter, total_loss, total_processed = 0, 0, 0, 0

                    for batch in train_loader:

                        images, labels, _ = batch
                        labels = labels.to(device='cuda')
                        images = images.to(device='cuda')

                        predictions = detector(images)
                        loss = functional.cross_entropy(predictions, labels)

                        optimizer.zero_grad()
                        loss.backward()  # Calculate Gradients
                        optimizer.step()  # Update Weights
                        loss_value = loss.item()

                        num_correct = get_num_correct(predictions, labels)
                        total_correct += num_correct
                        batch_counter += 1
                        total_processed += batch_size

                        if total_processed % 200 == 0:
                            print('Loss: ', loss_value, 'Number Correct', num_correct, 'Accuracy on batch', num_correct / batch_size)
                            tb.add_scalar('Loss', loss_value, batch_counter)
                            tb.add_scalar('Number Correct', num_correct, batch_counter)
                            tb.add_scalar('Accuracy on batch', num_correct / batch_size, batch_counter)

                print('Accuracy total', total_correct / len(train_ds))
                tb.add_scalar('Accuracy total', total_correct / len(train_ds))
                tb.close()

                torch.save(detector.state_dict(), f'models/model_lr:{lr}_bs:{batch_size}.pth')





    # model = Model(arguments)
    # model.load_state_dict(torch.load(PATH))
    # model.eval() // if we want to evaluate and not train again
