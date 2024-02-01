import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to('cuda')
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to('cuda')
    return tensor * std + mean


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
    def __init__(self, n_steps=1, step_size=0.2, sigma=0.1 ,device='cuda'):
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size

        self.device =device
        self.sigma = sigma
        self.normalizer = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    def get_grad(self, input, s_out, t_out, sample_label):
        grad_s = torch.autograd.grad(s_out[:, sample_label].sum(), input, allow_unused=True)[0]
        grad_y = torch.autograd.grad(t_out[:, sample_label].sum(), input, allow_unused=True)[0]
        # TODO (after the first stage): add entropy or some other things to improve quality
        return grad_y.detach() - grad_s.detach()

    def step(self, input, s_model, t_model, sample_label,decay=0):
        if decay!=0:
            cur_step_size=self.step_size/10
        else:
            cur_step_size=self.step_size

        for i in range(self.n_steps):
            tmp_input = input.clone().requires_grad_()

            s_output = s_model(tmp_input)
            t_output = t_model(tmp_input)

            grad = self.get_grad(tmp_input, s_output, t_output, sample_label).clamp_(-0.1, 0.1)

            import math
            tmp_input = tmp_input + grad * cur_step_size / 2 + math.sqrt(cur_step_size) * torch.randn(tmp_input.shape).to(self.device).normal_(0, 0.01)

            transformed_images = []
            for img in input:
                # pil_img = transforms.ToPILImage()(img)
                transformed_img = self.normalizer(img)
                transformed_images.append(transformed_img.unsqueeze(0))

            input = torch.cat(transformed_images).to(self.device)

        return tmp_input


def get_samples(t_model, s_model, train_loader, class_num=100, sample_num_per_class=1000,
                threshold=0.5, input_size=(64, 3, 32, 32), steps=64, decay=0,device='cuda'):

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

            img_per_step = [] # record the result in every step
            for _ in range(steps):
                input = sampler.step(input, s_model, t_model, cls)
                img_per_step.append(input.clone().detach())

                """
                # debug
                if _ % 8 == 0:
                    with torch.no_grad():
                        s_output = s_model(input)
                        t_output = t_model(input)
                        s_out = s_output.softmax(dim=-1)
                        t_out = t_output.softmax(dim=-1)
                        # print(s_output[:1, cls], t_output[:1, cls])
                        print(_, s_out[:10, cls], t_out[:10, cls])
                        #print(_, s_out[:1].argmin(), s_out[:1].min(), t_out[:1].argmax(), t_out[:1].max())
                """
            #decay step_size
            if decay!=0:
                for sps in range(steps):
                    input=sampler.step(input,s_model,t_model,cls,1)
                    """
                    # debug
                    if sps % 8 == 0:
                        with torch.no_grad():
                            s_output = s_model(input)
                            t_output = t_model(input)
                            s_out = s_output.softmax(dim=-1)
                            t_out = t_output.softmax(dim=-1)
                            # print(s_output[:1, cls], t_output[:1, cls])
                            print(_, s_out[:10, cls], t_out[:10, cls])
                            #print(_, s_out[:1].argmin(), s_out[:1].min(), t_out[:1].argmax(), t_out[:1].max())
                    """
                    
            
            #

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
                sample_pairs.append((abs(t_out[i, cls].item() - s_out[i, cls].item()), input[i]))

        sorted_pairs = sorted(sample_pairs, reverse=True)
        sorted_pairs = sorted_pairs[:int(sample_num_per_class * threshold)]
        for val, img in sorted_pairs:
            genrated_data.append((img.cpu().detach(), cls))

    return genrated_data
