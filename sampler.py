import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to('cuda')
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to('cuda')
    return tensor * std + mean


def random_sample(input, input_size):
    new_input = torch.zeros_like(input)
    random_indices = torch.randint(0, 4, (input_size[0], input_size[2], input_size[3]))
    #random_indices = torch.randint(0, input_size[0], (input_size[0], input_size[2], input_size[3]))
    for i in range(input_size[0]):
        for h in range(input_size[2]):
            for w in range(input_size[3]):
                selected_image_idx = random_indices[i, h, w]
                new_input[i, 0, h, w] = input[selected_image_idx, 0, h, w]
                new_input[i, 1, h, w] = input[selected_image_idx, 1, h, w]
                new_input[i, 2, h, w] = input[selected_image_idx, 2, h, w]

    normalizer = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transformed_images = []
    for img in new_input:
        # pil_img = transforms.ToPILImage()(img)
        transformed_img = normalizer(img)
        transformed_images.append(transformed_img.unsqueeze(0))

    new_input = torch.cat(transformed_images).to('cuda')
    return new_input


def output(s_out, t_out, input): # visualize
    for i in range(input.shape[0]):
        normalized_image = input[i]
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        original_image = (denormalize(normalized_image, mean, std) * 255).to(torch.uint8)
        pil_image = to_pil_image(original_image)

        os.system("mkdir -p ./save_image/")
        pil_image.save(f'./save_image/img_{i}_s_{int(s_out[i,0].item()*100)}_t_{int(t_out[i,0].item()*100)}.png')

    print("Finished!")


class LangevinSampler(nn.Module):
    def __init__(self, n_steps=1, step_size=0.2, sigma=0.1, device='cuda'):
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size

        self.device =device
        self.sigma = sigma
        self.normalizer = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    def get_grad(self, input, s_out, t_out, sample_label):
        grad_s = torch.autograd.grad(s_out[:, sample_label].sum(), input, allow_unused=True)[0]
        grad_t = torch.autograd.grad(t_out[:, sample_label].sum(), input, allow_unused=True, retain_graph=True)[0]
        probs = F.softmax(t_out, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1).sum()
        grad_entropy = torch.autograd.grad(entropy, input, allow_unused=True)[0]
        # TODO (after the first stage): add entropy or some other things to improve quality
        # print(grad_t, grad_s, grad_entropy)
        return grad_t.detach() - grad_s.detach() + grad_entropy.detach()

    def step(self, input, s_model, t_model, sample_label):

        for i in range(self.n_steps):
            tmp_input = input.clone().requires_grad_()

            s_output = s_model(tmp_input)
            t_output = t_model(tmp_input)

            grad = self.get_grad(tmp_input, s_output, t_output, sample_label).clamp_(-0.1, 0.1)

            import math
            tmp_input = tmp_input + grad * self.step_size / 2 + math.sqrt(self.step_size) * torch.randn(tmp_input.shape).to(self.device).normal_(0, 0.01)

            transformed_images = []
            for img in input:
                # pil_img = transforms.ToPILImage()(img)
                transformed_img = self.normalizer(img)
                transformed_images.append(transformed_img.unsqueeze(0))

            input = torch.cat(transformed_images).to(self.device)

        return tmp_input


def get_samples(t_model, s_model, train_loader, class_num=100, sample_num_per_class=1000,
                threshold=0.5, input_size=(64, 3, 32, 32), steps=64, device='cuda'):

    # the final num of samples should be sample_num_per_class * class_num * threshold
    # the output is constructed as a 1-dim array and every element is a tuple (img, label)
    # in sample.py please make sure you give the same input_size as the original data!!!

    print("Start sampling...")
    s_model.eval()
    t_model.eval()
    sampler = LangevinSampler(n_steps=1, sigma=1e-1, step_size=1e-2, device=device)
    genrated_data = []

    for cls in tqdm(range(class_num)):
        cnt = 0
        sample_pairs = []
        while cnt < sample_num_per_class:
            cnt += input_size[0]

            input, _, _ = next(iter(train_loader[cls]))
            if False:
                input = random_sample(input, input_size)

            input = input.to(device)

            """
            # create initial noisy picture
            # input = torch.randint(0, 256, input_size, device=device, dtype=torch.uint8)
            input = torch.randn(input_size, device=device)
            normalizer = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

            transformed_images = []
            for img in input:
                # pil_img = transforms.ToPILImage()(img)
                transformed_img = normalizer(img)
                transformed_images.append(transformed_img.unsqueeze(0))

            input = torch.cat(transformed_images).to(device)
            """

            # start sampling
            torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary

            # img_per_step = []
            for _ in range(steps):
                input = sampler.step(input, s_model, t_model, cls)
                # img_per_step.append(input.clone().detach())

                #"""
                # debug
                if _ % 8 == 0:
                    with torch.no_grad():
                        s_output = s_model(input)
                        t_output = t_model(input)
                        s_out = s_output.softmax(dim=-1)
                        t_out = t_output.softmax(dim=-1)
                        # print(s_output[:1, cls], t_output[:1, cls])
                        # print(_, s_out[:10, cls], t_out[:10, cls])
                        #print(_, s_out[:1].argmin(), s_out[:1].min(), t_out[:1].argmax(), t_out[:1].max())
                #"""

            with torch.no_grad():
                s_output = s_model(input)
                t_output = t_model(input)
                s_out = s_output.softmax(dim=-1)
                t_out = t_output.softmax(dim=-1)

            #print(s_out[:10, cls], t_out[:10, cls])

            if cls == 0 and cnt == input_size[0]:
                output(s_out, t_out, input)

            for i in range(input_size[0]):
                # TODO (after the first stage): add entropy or some other things to improve quality
                entropy = -torch.sum(t_out[i] * torch.log(t_out[i]))
                # print(abs(t_out[i, cls].item() - s_out[i, cls].item()), entropy)
                sample_pairs.append((0 * abs(t_out[i, cls].item() - s_out[i, cls].item()) + entropy.item(), cnt-input_size[0]+i, input[i]))

        sorted_pairs = sorted(sample_pairs, reverse=True)
        sorted_pairs = sorted_pairs[:int(sample_num_per_class * threshold)]
        for val, idx, img in sorted_pairs:
            genrated_data.append((img.cpu().detach(), cls))

    return genrated_data
