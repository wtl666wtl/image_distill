from sampler import get_samples
import argparse


def load_model(path):
    # TODO
    ...


def store_data(data, output_path):
    # TODO
    ...


def main(args):
    # TODO: first load teacher and student from the path
    # you can first try to solve one t/s pair like (t: wrn-40-2, s: wrn-16-2)
    # I think it's not hard for you
    t_model = load_model(args.t_model)
    t_model = t_model.to('cuda')
    s_model = load_model(args.s_model)
    s_model = s_model.to('cuda')

    # call sampler to generate data, I will write the sampler
    data = get_samples(t_model, s_model)

    # last step: store data into the specific path
    store_data(data, args.output_path)


if __name__ == '__main__':
    # TODO: feel free to add more args here if you want
    parser = argparse.ArgumentParser(description='data sampling')
    parser.add_argument('--t_model', default='...',
                        help='teacher model path')
    parser.add_argument('--s_model', default='...',
                        help='student model path')
    parser.add_argument('--output_path', default='...',
                        help='generated data path')

    args = parser.parse_args()
    main(args)