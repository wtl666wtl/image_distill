import torch
import torch.nn as nn


class LangevinSampler(nn.Module):
    def __init__(self, n_steps=10, sample_label=1, step_size=0.2, sigma=0.1, device='cuda'):
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size

        self.sample_label = sample_label
        self.device =device
        self.sigma = sigma

    def get_grad(self, x, y, z):
        gxy = torch.autograd.grad(y.logits[:, self.sample_label].sum(), x, allow_unused=True)[0]
        gxz = torch.autograd.grad(z.logits[:, 1 - self.sample_label].sum(), x, allow_unused=True)[0]
        return gxy.detach() + gxz.detach()

    def step(self, input_ids, s_model, t_model):

        # TODO: fix some details...
        for i in range(self.n_steps):
            embeds = input_ids.clone().requires_grad_()
            output = s_model(inputs_embeds=embeds, labels=(self.sample_label * torch.ones(input_ids.shape[0]).type(torch.LongTensor)).cuda())
            t_output = t_model(inputs_embeds=embeds, labels=(self.sample_label * torch.ones(input_ids.shape[0]).type(torch.LongTensor)).cuda())
            # print(output.loss)
            grad = self.get_grad(embeds, output, t_output).clamp_(-0.1, 0.1)

            embeds = embeds + grad * self.step_size / 2
            # + math.sqrt(self.step_size) * torch.randn(embeds.shape).to(self.device).normal_(0, 0.3)

        return embeds


def get_samples(t_model, s_model, steps=32, sample_num=16, sample_label=1, threshold=0.2, device='cuda'):
    print("Start sampling...")
    s_model.eval()

    # TODO: rewrite it...
    ...