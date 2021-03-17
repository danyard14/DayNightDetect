from torch.utils.tensorboard import SummaryWriter
from day_night_detector import DayNightDetector
from data_loader import ImagesDataset
from torch.utils.data import DataLoader
from get_data import get_all_with_field
from torch import optim
import torch
from torch.nn import functional

from test import get_wrong_classified_images

scene_detection_col = 'scene_detection'
learning_rates = [0.001, 0.0001]
total_loss = 0
total_correct = 0
data_folder = '/home/student/Desktop/Dan/day_night_detection'
batch_sizes = [1, 10, 20, 50, 100]
epochs = 2


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


if __name__ == '__main__':
    print('starting training')

    train_data = get_all_with_field(scene_detection_col, field_name='train_test', field_value='train')
    train_ds = ImagesDataset(train_data)

    val_data = get_all_with_field(scene_detection_col, field_name='train_test', field_value='val')
    val_ds = ImagesDataset(val_data)

    for lr in learning_rates:
        print('lr=', lr)
        for batch_size in batch_sizes:
            print('batch_size=', batch_size)

            detector = DayNightDetector()
            detector = detector.to(device='cuda')

            # train
            detector.train()
            optimizer = optim.Adam(detector.parameters(), lr=lr)
            train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
            tb = SummaryWriter(comment=f'train batch_size={batch_size} lr={lr}')

            # for epoch in range(epochs):
            #     print('train epoch', epoch)
            #     total_correct, batch_counter, total_loss = 0, 0, 0
            #     for batch in train_loader:
            #         print('\t batch', str(batch_counter) + ':')
            #         batch_counter += 1
            #
            #         images, labels, _ = batch
            #
            #         labels = labels.to(device='cuda')
            #         images = images.to(device='cuda')
            #
            #         preds = detector(images)  # Pass Batch
            #         loss = functional.cross_entropy(preds, labels)  # Calculate Loss
            #
            #         optimizer.zero_grad()
            #         loss.backward()  # Calculate Gradients
            #         optimizer.step()  # Update Weights
            #
            #         total_loss += loss.item()
            #         total_correct += get_num_correct(preds, labels)
            #         print('\t loss: ', loss.item())
            #
            #     tb.add_scalar('Loss', total_loss, epoch)
            #     tb.add_scalar('Number Correct', total_correct, epoch)
            #     tb.add_scalar('Accuracy', total_correct / len(val_ds), epoch)
            #     tb.close()

            # validation
            tb = SummaryWriter(comment=f'validation batch_size={batch_size} lr={lr}')
            detector.eval()
            total = 0
            wrong = 0
            wrong_examples = []
            val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
            for epoch in range(epochs):
                for batch in val_loader:
                    print(type(batch))
                    print(type(batch[0]), type(batch[1]), type(batch[2]))
                    images, labels, _ = batch
                    total += len(images)
                    labels = labels.to(device='cuda')
                    images = images.to(device='cuda')

                    preds = detector(images)

                    wc = get_wrong_classified_images(preds, labels)

                    wrong += len(wc)
                    wrong_examples.extend(wc)

                tb.add_scalar('Loss', total_loss, epoch)
                tb.add_scalar('Number Correct', total_correct, epoch)
                tb.add_scalar('Accuracy', total_correct / len(val_ds), epoch)
                tb.close()

            torch.save(detector.state_dict(), f'models/model_{lr}_{batch_size}.pth')

    # model = Model(arguments)
    # model.load_state_dict(torch.load(PATH))
    # model.eval() // if we want to evaluate and not train again
