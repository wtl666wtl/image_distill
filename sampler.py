import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss


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
<<<<<<< HEAD
    def __init__(self, n_steps=1, step_size=0.2, sigma=0.1 ,device='cuda'):
=======
    def __init__(self, args, n_steps=1, step_size=0.2, device='cuda'):
>>>>>>> de595eb39139302552c50ed4b718dd2b4907e085
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size

        self.device =device
        self.distill = args.distill
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta
        self.normalizer = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(args.kd_T)
        if self.distill == 'kd':
            criterion_kd = DistillKL(args.kd_T)
        elif self.distill == 'hint':
            criterion_kd = HintLoss()
            regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
            module_list.append(regress_s)
            trainable_list.append(regress_s)
        elif self.distill == 'crd':
            opt.s_dim = feat_s[-1].shape[1]
            opt.t_dim = feat_t[-1].shape[1]
            opt.n_data = n_data
            criterion_kd = CRDLoss(opt)
            module_list.append(criterion_kd.embed_s)
            module_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_s)
            trainable_list.append(criterion_kd.embed_t)
        elif self.distill == 'attention':
            criterion_kd = Attention()
        elif self.distill == 'nst':
            criterion_kd = NSTLoss()
        elif self.distill == 'similarity':
            criterion_kd = Similarity()
        elif self.distill == 'rkd':
            criterion_kd = RKDLoss()
        elif self.distill == 'pkt':
            criterion_kd = PKT()
        elif self.distill == 'kdsvd':
            criterion_kd = KDSVD()
        elif self.distill == 'correlation':
            criterion_kd = Correlation()
            embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
            embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
            module_list.append(embed_s)
            module_list.append(embed_t)
            trainable_list.append(embed_s)
            trainable_list.append(embed_t)
        elif self.distill == 'vid':
            s_n = [f.shape[1] for f in feat_s[1:-1]]
            t_n = [f.shape[1] for f in feat_t[1:-1]]
            criterion_kd = nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
            # add this as some parameters in VIDLoss need to be updated
            trainable_list.append(criterion_kd)
        elif self.distill == 'abound':
            s_shapes = [f.shape for f in feat_s[1:-1]]
            t_shapes = [f.shape for f in feat_t[1:-1]]
            connector = Connector(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(connector)
            init_trainable_list.append(model_s.get_feat_modules())
            criterion_kd = ABLoss(len(feat_s[1:-1]))
            init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
            # classification
            module_list.append(connector)
        elif self.distill == 'factor':
            s_shape = feat_s[-2].shape
            t_shape = feat_t[-2].shape
            paraphraser = Paraphraser(t_shape)
            translator = Translator(s_shape, t_shape)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(paraphraser)
            criterion_init = nn.MSELoss()
            init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
            # classification
            criterion_kd = FactorTransfer()
            module_list.append(translator)
            module_list.append(paraphraser)
            trainable_list.append(translator)
        elif self.distill == 'fsp':
            s_shapes = [s.shape for s in feat_s[:-1]]
            t_shapes = [t.shape for t in feat_t[:-1]]
            criterion_kd = FSP(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(model_s.get_feat_modules())
            init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
            # classification training
            pass
        else:
            raise NotImplementedError(self.distill)

        self.criterion_list = nn.ModuleList([])
        self.criterion_list.append(criterion_cls)  # classification loss
        self.criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        self.criterion_list.append(criterion_kd)  # other knowledge distillation loss


    def get_grad_old(self, input, s_out, t_out, sample_label): # fixed grad_diff (old version)
        grad_s = torch.autograd.grad(s_out[:, sample_label].sum(), input, allow_unused=True)[0]
        grad_t = torch.autograd.grad(t_out[:, sample_label].sum(), input, allow_unused=True, retain_graph=True)[0]
        probs = F.softmax(t_out, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1).sum()
        grad_entropy = torch.autograd.grad(entropy, input, allow_unused=True)[0]
        # print(grad_t, grad_s, grad_entropy)
        return grad_t.detach() - grad_s.detach() + grad_entropy.detach()


    def get_grad(self, input, model_s, model_t, sample_label):
        preact = False
        if self.distill in ['abound']:
            preact = True

        model_t.zero_grad()
        model_s.zero_grad()
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        feat_t, logit_t = model_t(input, is_feat=True, preact=preact)

        criterion_cls = self.criterion_list[0]
        criterion_div = self.criterion_list[1]
        criterion_kd = self.criterion_list[2]

        target = sample_label * torch.ones(logit_t.shape[0], dtype=torch.long).to(self.device)
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if self.distill == 'kd':
            loss_kd = 0
        elif self.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif self.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif self.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif self.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif self.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif self.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(self.distill)

        loss = self.gamma * loss_cls + self.alpha * loss_div + self.beta * loss_kd

        print(loss)

        grad_diff = torch.autograd.grad(loss, input, allow_unused=True)[0]#, retain_graph=True)[0]

        # probs = F.softmax(logit_t, dim=1)
        # entropy = -torch.sum(probs * torch.log(probs), dim=1).sum()
        # grad_entropy = torch.autograd.grad(entropy, input, allow_unused=True)[0]
        return grad_diff.detach() # + grad_entropy.detach()


    def step(self, input, s_model, t_model, sample_label,decay=0):
        if decay!=0:
            cur_step_size=self.step_size/10
        else:
            cur_step_size=self.step_size

        for i in range(self.n_steps):
            tmp_input = input.clone().requires_grad_()

            # s_output = s_model(tmp_input)
            # t_output = t_model(tmp_input)

            grad = self.get_grad(tmp_input, s_model, t_model, sample_label).clamp_(-0.1, 0.1)

            import math
            tmp_input = tmp_input + grad * cur_step_size / 2 + math.sqrt(cur_step_size) * torch.randn(tmp_input.shape).to(self.device).normal_(0, 0.01)

            transformed_images = []
            for img in input:
                # pil_img = transforms.ToPILImage()(img)
                transformed_img = self.normalizer(img)
                transformed_images.append(transformed_img.unsqueeze(0))

            input = torch.cat(transformed_images).to(self.device)

        return tmp_input


<<<<<<< HEAD
def get_samples(t_model, s_model, train_loader, class_num=100, sample_num_per_class=1000,
                threshold=0.5, input_size=(64, 3, 32, 32), steps=64, decay=0,device='cuda'):
=======
def get_samples(t_model, s_model, train_loader, args, class_num=100, sample_num_per_class=1000,
                threshold=0.5, input_size=(64, 3, 32, 32), steps=64, device='cuda'):
>>>>>>> de595eb39139302552c50ed4b718dd2b4907e085

    # the final num of samples should be sample_num_per_class * class_num * threshold
    # the output is constructed as a 1-dim array and every element is a tuple (img, label)
    # in sample.py please make sure you give the same input_size as the original data!!!

    print("Start sampling...")
    s_model.eval()
    t_model.eval()
    sampler = LangevinSampler(args, n_steps=1, step_size=5e-2, device=device)
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
<<<<<<< HEAD
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
=======
                        # print(_, s_out[:10, cls], t_out[:10, cls])
                        # print(_, s_out[:1].argmin(), s_out[:1].min(), t_out[:1].argmax(), t_out[:1].max())
                #"""
>>>>>>> de595eb39139302552c50ed4b718dd2b4907e085

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

        print(f"cls {cls}: {val}")

    return genrated_data
