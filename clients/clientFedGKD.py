import copy
import torch
from torch import optim

from clients.clientBase import ClientBase
from data.utils import CONFIGS


class ClientFedGKD(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientFedGKD, self).__init__(args, client_id, data)
        self.teacher = copy.deepcopy(self.model)
        self.student = copy.deepcopy(self.model)
        self.optimizer_s = optim.SGD(
            self.student.parameters(), lr=CONFIGS[args.dataset]['lr'], momentum=CONFIGS[args.dataset]['momentum'],
            weight_decay=CONFIGS[args.dataset]['weight_decay']
        )

    def train_with_distill(self, teacher_weight, student_weight, num_batches_per_epoch, gamma=0.2):
        self.teacher.load_state_dict(teacher_weight)
        self.teacher.eval()
        self.student.load_state_dict(student_weight)
        self.student.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss_ce = []
        batch_loss_kl = []
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                student_outputs = self.student(images)
                loss_ce = self.ce(student_outputs['logits'], classes)
                with torch.no_grad():
                    teacher_outputs = self.teacher(images)
                x = student_outputs['logits'].log_softmax(dim=-1)
                y = teacher_outputs['logits'].softmax(dim=-1)
                loss_kl = self.kl(x, y.detach())
                total_loss = loss_ce + loss_kl * gamma / 2
                self.optimizer_s.zero_grad()
                total_loss.backward()
                self.optimizer_s.step()
                batch_loss_ce.append(loss_ce.cpu().item())
                batch_loss_kl.append(loss_kl.cpu().item())
        avg_loss_ce = sum(batch_loss_ce) / len(batch_loss_ce)
        avg_loss_kl = sum(batch_loss_kl) / len(batch_loss_kl)
        print(' [ce_loss: %.5f, kl_loss: %.5f]' % (avg_loss_ce, avg_loss_kl))

        return copy.deepcopy(self.student.state_dict())

    def train_with_distill_vote(self, teacher_weights, student_weight, num_batches_per_epoch, gamma=0.2):
        teachers = []
        for teacher_weight in teacher_weights:
            teacher = copy.deepcopy(self.teacher)
            teacher.load_state_dict(teacher_weight)
            teacher.eval()
            teachers.append(teacher)
        student = self.student
        student.load_state_dict(student_weight)
        student.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss_ce = []
        batch_loss_kl = []
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                student_outputs = student(images)
                loss_ce = self.ce(student_outputs['logits'], classes)
                teacher_logits = []
                for teacher in teachers:
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                    teacher_logits.append(teacher_outputs['logits'])
                avg_teacher_logit = sum(teacher_logits) / len(teacher_logits)
                x = student_outputs['logits'].log_softmax(dim=-1)
                y = avg_teacher_logit.softmax(dim=-1)
                loss_kl = self.kl(x, y.detach())
                total_loss = loss_ce + loss_kl * gamma / 2
                self.optimizer_s.zero_grad()
                total_loss.backward()
                self.optimizer_s.step()
                batch_loss_ce.append(loss_ce.cpu().item())
                batch_loss_kl.append(loss_kl.cpu().item())
        avg_loss_ce = sum(batch_loss_ce) / len(batch_loss_ce)
        avg_loss_kl = sum(batch_loss_kl) / len(batch_loss_kl)
        print(' [CE Loss: %.5f, KL Loss:%.5f]' % (avg_loss_ce, avg_loss_kl))

        return copy.deepcopy(student.state_dict())
