from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as t
import glob
from torch import tensor
from PIL import Image

image_size = 64  # TODO: define this
data_path = r'C:\Users\danya\Desktop\day_night_detection'


class ImagesDataset(Dataset):
    def __init__(self, path_to_data: str):
        super(ImagesDataset, self).__init__()
        day_images = glob.glob(f'{path_to_data}/**/day/**.jpg', recursive=True)
        night_images = glob.glob(f'{path_to_data}/**/night/**.jpg', recursive=True)
        data = day_images
        data.extend(night_images)
        data = [Image.open(p) for p in data]
        labels = [0 for _ in range(len(day_images))]
        labels.extend([1 for _ in range(len(night_images))])
        self.data = data
        self.labels = labels
        self.transform = t.Compose([t.Resize(image_size), t.ToTensor()])

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.data[index]
        label = self.labels[index]
        item = self.transform(image)
        return item, tensor(label)


if __name__ == '__main__':
    ds = ImagesDataset(data_path)
    transformed_dataset = ImagesDataset(path_to_data=data_path)
    train_dl = DataLoader(transformed_dataset, 100, shuffle=True, num_workers=3, pin_memory=True)
    for images in train_dl:
        print(images)
