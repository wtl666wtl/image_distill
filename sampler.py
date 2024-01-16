import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class LangevinSampler(nn.Module):
    def __init__(self, n_steps=1, step_size=0.2, sigma=0.1, device='cuda'):
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size

        self.device =device
        self.sigma = sigma

    def get_grad(self, input, s_out, t_out, sample_label):
        grad_s = torch.autograd.grad(s_out[:, sample_label].sum(), input, allow_unused=True)[0]
        grad_y = torch.autograd.grad(t_out[:, sample_label].sum(), input, allow_unused=True)[0]
        # TODO (after the first stage): add entropy or some other things to improve quality
        return grad_y.detach() - grad_s.detach()

    def step(self, input, s_model, t_model, sample_label):

        for i in range(self.n_steps):
            tmp_input = input.clone().requires_grad_()

            s_output = s_model(tmp_input)
            t_output = t_model(tmp_input)

            grad = self.get_grad(tmp_input, s_output, t_output, sample_label).clamp_(-0.1, 0.1)

            tmp_input = tmp_input + grad * self.step_size / 2
            # + math.sqrt(self.step_size) * torch.randn(embeds.shape).to(self.device).normal_(0, 0.3)

        return tmp_input


def get_samples(t_model, s_model, class_num=100, sample_num_per_class=10000,
                threshold=0.5, input_size=(128, 3, 32, 32), steps=64, device='cuda'):

    # the final num of samples should be sample_num_per_class * class_num * threshold
    # the output is constructed as a 2-dim array [class_num, sample_num_per_class * threshold]
    # in sample.py please make sure you give the same input_size as the original data!!!

    print("Start sampling...")
    s_model.eval()
    sampler = LangevinSampler(n_steps=1, sigma=1e-1, step_size=5e-4, device=device)
    genrated_data = []

    for cls in range(class_num):
        cnt = 0
        sample_pairs = []
        while cnt < sample_num_per_class:
            cnt += input_size[0]

            # create initial noisy picture
            input = torch.randint(0, 256, input_size, device=device, dtype=torch.uint8)
            # TODO: add this line if having bugs
            # input = transforms.ToPILImage()(input)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            input = train_transform(input)

            # start sampling
            torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary

            img_per_step = [] # record the result in every step
            for _ in range(steps):
                input = sampler.step(input, s_model, t_model, cls)
                img_per_step.append(input.clone().detach())

            # TODO (not important): print the output for debugging
            with torch.no_grad():
                s_output = s_model(input)
                t_output = t_model(input)
                s_out = s_output.softmax(dim=-1)
                t_out = t_output.softmax(dim=-1)

            for i in range(input_size[0]):
                # TODO (after the first stage): add entropy or some other things to improve quality
                sample_pairs.append((abs(t_out[i, cls].item() - s_out[i, cls].item()), input[i]))

        sorted_pairs = sorted(sample_pairs, reverse=True)
        sorted_pairs = sorted_pairs[:sample_num_per_class * threshold]
        for val, img in sorted_pairs:
            genrated_data.append((img.cpu().detach(), cls))

    return genrated_data
