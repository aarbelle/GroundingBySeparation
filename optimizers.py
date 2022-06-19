import torch.optim as optim


def get_optimizer(model, params):
    if hasattr(model.image_model, 'backbone'):
        backbone_parameters = list(model.image_model.backbone.parameters())
        all_model_params = [m for n, m in model.named_parameters() if 'backbone.' not in n and 'caption_model.model.' not in n]
        model_params = [{'params': all_model_params},
                        {'params': backbone_parameters, 'lr': params.backbone_lr}]
    else:
        model_params = [m for n, m in model.named_parameters() if 'caption_model.model.' not in n]
    if params.optimizer == 'adam':
        return optim.Adam(model_params, lr=params.learning_rate, weight_decay=params.weight_decay, betas=params.beta)
    elif params.optimizer == 'sgd':
        return optim.SGD(model_params, lr=params.learning_rate, weight_decay=params.weight_decay,
                         momentum=params.momentum)
    elif params.optimizer == 'rmsprop':
        return optim.RMSprop(model_params, lr=params.learning_rate, weight_decay=params.weight_decay,
                             momentum=params.momentum)
    else:
        raise NotImplementedError('Optimizer {} not implemented in code'.format(params.optimizer))
