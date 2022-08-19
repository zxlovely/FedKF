import copy
import torch
from torch import optim

from clients.clientBase import ClientBase
from data.utils import CONFIGS


class ClientFedKF(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientFedKF, self).__init__(args, client_id, data)
        self.teacher = copy.deepcopy(self.model)
        self.student = copy.deepcopy(self.model)
        self.generator = None
        self.optimizer_s = optim.SGD(
            self.student.parameters(), lr=CONFIGS[args.dataset]['lr'], momentum=CONFIGS[args.dataset]['momentum'],
            weight_decay=CONFIGS[args.dataset]['weight_decay']
        )
        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=1e-3)
        self.g_use_kl = CONFIGS[args.dataset]['g_use_kl']
        self.generator_weights = []
        self.teacher_weights = []

    def train_with_distill(self, teacher_weight, student_weight, oh=0.1, a=0.1, gamma=1, num_batches_per_epoch=31):
        self.teacher.load_state_dict(teacher_weight)
        self.student.load_state_dict(student_weight)
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss_ce = []
        batch_loss_kl = []
        batch_loss_ie = []
        batch_loss_oh = []
        batch_loss_a = []
        ps = []
        fake_classes = []
        self.generator.train()
        self.teacher.eval()
        self.student.train()
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                # train local model
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.student(images)
                loss_ce = self.ce(outputs['logits'], classes)
                noises = torch.randn(self.args.batch_size_local_training, self.generator.noise_dim,
                                     device=self.args.device)
                fakes = self.generator(noises)
                student_outputs = self.student(fakes.detach())
                teacher_outputs = self.teacher(fakes)
                loss_kl = self.kl(
                    student_outputs['logits'].log_softmax(dim=1),
                    teacher_outputs['logits'].softmax(dim=1).detach()
                )
                loss_s = loss_ce + loss_kl * gamma
                self.optimizer_s.zero_grad()
                loss_s.backward()
                self.optimizer_s.step()
                batch_loss_ce.append(loss_ce.cpu().item())
                batch_loss_kl.append(loss_kl.cpu().item())
                '''train generator'''
                classes_g = teacher_outputs['logits'].argmax(dim=1)
                fake_classes.append(classes_g)
                p = teacher_outputs['logits'].softmax(dim=1).mean(dim=0)
                ps.append(p.detach())
                loss_oh = self.ce(teacher_outputs['logits'], classes_g)
                loss_ie = 1 + (p * p.log() / torch.log(torch.ones_like(p) * self.num_classes)).sum()
                loss_a = - teacher_outputs['features'].abs().mean()
                loss_g = loss_ie + loss_oh * oh + loss_a * a

                # Train generator with kl loss
                if self.g_use_kl:
                    student_outputs = self.student(fakes)
                    loss_kl = self.kl(
                        student_outputs['logits'].log_softmax(dim=1),
                        teacher_outputs['logits'].softmax(dim=1).detach()
                    )
                    loss_g -= loss_kl * gamma

                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
                batch_loss_ie.append(loss_ie.cpu().item())
                batch_loss_oh.append(loss_oh.cpu().item())
                batch_loss_a.append(loss_a.cpu().item())

        avg_loss_ie = sum(batch_loss_ie) / len(batch_loss_ie)
        avg_loss_oh = sum(batch_loss_oh) / len(batch_loss_oh)
        avg_loss_a = sum(batch_loss_a) / len(batch_loss_a)
        avg_loss_ce = sum(batch_loss_ce) / len(batch_loss_ce)
        avg_loss_kl = sum(batch_loss_kl) / len(batch_loss_kl)
        print(' [ie_loss: %.5f, oh_loss: %.5f, a_loss: %.5f]' % (avg_loss_ie, avg_loss_oh, avg_loss_a), end='')
        print(' [ce_loss: %.5f, kl_loss: %.5f]' % (avg_loss_ce, avg_loss_kl), end='')
        mean_p = (sum(ps) / len(ps)).cpu().numpy().tolist()
        fake_classes = torch.cat(fake_classes, dim=0).cpu().numpy().tolist()
        nums_per_class = [0 for _ in range(len(mean_p))]
        for cla in fake_classes:
            nums_per_class[cla] += 1
        print(' [probability: (%.3f, %.3f, %.3f), ' % (min(mean_p), sum(mean_p) / len(mean_p), max(mean_p)), end='')
        print(f'samples: ({min(nums_per_class)}, {sum(nums_per_class) // len(nums_per_class)}, {max(nums_per_class)})]')

        return copy.deepcopy(self.student.state_dict())
