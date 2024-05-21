from torchvision import transforms
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

inverse_normalize = transforms.Normalize(
    mean=(-0.5 / 0.5,),
    std=(1.0 / 0.5,)
)