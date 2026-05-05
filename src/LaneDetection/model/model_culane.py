import torch
from .backbone import resnet


def _init_weights(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0.0, std=0.01)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            _init_weights(child)


class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row=None, num_cls_row=None,
                 num_grid_col=None, num_cls_col=None, num_lane_on_row=None, num_lane_on_col=None,
                 use_aux=False, input_height=None, input_width=None, fc_norm=False):
        super().__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ['34', '18', '34fca'] else torch.nn.Conv2d(2048, 8, 1)
        _init_weights(self.cls)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        fea = self.pool(fea)
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        return {
            'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
        }