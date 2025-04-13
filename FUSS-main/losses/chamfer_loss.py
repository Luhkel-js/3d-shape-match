import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance,  mesh_edge_loss

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ChamferLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, mesh_x, mesh_y):
        loss, _ = chamfer_distance(mesh_x, mesh_y)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, verts, faces):
        mesh = Meshes(verts=[verts], faces=[faces])
        loss = mesh_edge_loss(mesh)
        return self.loss_weight * loss