import timm


def get_model(model_name):

    return timm.create_model(model_name, features_only=True, pretrained=True)


def get_model_info(model):
    model_downscales = []
    model_depths = model.feature_info.channels()
    current_downscale = 1

    for d in model.feature_info.reduction():
        model_downscales.append(int(d/current_downscale))
        current_downscale = d
    return model_depths, model_downscales
