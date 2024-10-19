import segmentation_models_pytorch

class UnetModel:
    def __init__(self, encoder_name = 'resnet 34', encoder_weights='imagenet', in_channels=3, num_cls=1):
        self.model = segmentation_models_pytorch.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_cls,
            activation=None
        )
    
    def get_model(self):
        return self.model


class DeepLabV3PlusModel:
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, num_cls=1):
        self.model = segmentation_models_pytorch.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_cls,
            activation=None
        )

    def get_model(self):
        return self.model