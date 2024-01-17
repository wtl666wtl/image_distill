from sampler import get_samples
from models import model_dict
from torch.utils.data import Dataset
import argparse
import pickle
import torch


class ImgDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label, idx

def get_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def store_data(data, output_path):
    dataset = ImgDataset(data)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


def main(args):
    # TODO: first load teacher and student from the path
    # you can first try to solve one t/s pair like (t: wrn-40-2, s: wrn-16-2)
    t_model = load(args.t_model, args.class_num)
    t_model = t_model.to('cuda')
    s_model = load(args.s_model, args.class_num)
    s_model = s_model.to('cuda')

    # call sampler to generate data
    data = get_samples(t_model, s_model, args.class_num,
                       args.sampler_num_per_class, args.threshold,
                       args.input_size, args.steps)

    # store data into the specific path
    store_data(data, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data sampling')
    parser.add_argument('--path_t', default='...',
                        help='teacher model path')
    parser.add_argument('--path_s', default='...',
                        help='student model path')
    parser.add_argument('--output_path', default='add_data/cifar-100/train_dataset_add.pkl',
                        help='generated data path')
    parser.add_argument('--class_num', default=100, type=int,
                        help='number of classes')
    parser.add_argument('--sample_num_per_class', default=10000, type=int,
                        help='number of samples per class')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='only consider the samples with high values')
    parser.add_argument('--input_size', default=(128, 3, 32, 32), type=tuple,
                        help='input image size')
    parser.add_argument('--steps', default=64, type=int,
                        help='sampling steps')

    args = parser.parse_args()
    main(args)