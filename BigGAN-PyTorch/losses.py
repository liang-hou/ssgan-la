import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis


def loss_multi_hinge_dis(dis_fake, dis_real, label_fake, label_real):
  dis_fake_choose = torch.gather(dis_fake, -1, label_fake.view(-1, 1))
  dis_real_choose = torch.gather(dis_real, -1, label_real.view(-1, 1))
  
  loss_fake = F.relu(1. - dis_fake_choose + dis_fake)
  loss_real = F.relu(1. - dis_real_choose + dis_real)

  loss_fake = torch.masked_select(loss_fake, torch.eye(dis_fake.size(1), device=loss_fake.device)[label_fake] < 0.5).mean()
  loss_real = torch.masked_select(loss_real, torch.eye(dis_real.size(1), device=loss_real.device)[label_real] < 0.5).mean()

  return loss_real + loss_fake


def loss_multi_hinge_gen(dis_fake, label_fake):
  dis_pos_choose = torch.gather(dis_fake, -1, (label_fake - 1).view(-1, 1))
  loss_pos = dis_fake - dis_pos_choose
  loss_pos = torch.masked_select(loss_pos, torch.eye(dis_fake.size(1), device=loss_pos.device)[label_fake - 1] < 0.5).mean()

  dis_neg_choose = torch.gather(dis_fake, -1, (label_fake).view(-1, 1))
  loss_neg = dis_fake - dis_neg_choose
  loss_neg = torch.masked_select(loss_neg, torch.eye(dis_fake.size(1), device=loss_neg.device)[label_fake] < 0.5).mean()

  return loss_pos - loss_neg