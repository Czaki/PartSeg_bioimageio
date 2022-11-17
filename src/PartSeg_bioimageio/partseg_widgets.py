from weakref import ref

from magicgui import register_type
from magicgui.types import PathLike
from magicgui.widgets import Container, EmptyWidget, FileEdit, create_widget
from PartSegImage import Channel


class BioImageModel:
    def __init__(self, path: PathLike, channels):
        self.path = path
        self.channels = channels

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value["path"], PathLike):
            raise ValueError("path must be a PathLike")
        return cls(**value)

    def as_dict(self):
        return {"path": self.path, "channels": self.channels}


class BioImageWidget(Container):
    multiline = True

    def __init__(self, **kwargs):
        self.file_select = FileEdit(
            label="Model RDF",
            filter="*.yaml",
            tooltip="Path to model RDF file",
        )
        self.channel_select_li = []
        self.image = lambda: None
        if "value" in kwargs:
            self.file_select.value = kwargs["value"].path
        self.file_select.changed.connect(self.update_model)
        super().__init__(widgets=[self.file_select], scrollable=False)
        self.margins = (0, 0, 0, 0)
        # self.native.setMinimumHeight(200)

    def update_model(self):
        if (
            not self.file_select.value.exists()
            or not self.file_select.value.is_file()
        ):
            return
        channels_num = calc_number_of_channels(self.file_select.value)
        if len(self.channel_select_li) < channels_num:
            for i in range(len(self.channel_select_li), channels_num):
                w = create_widget(
                    annotation=Channel,
                    label=f"Channel {i}",
                    options={"tooltip": "Select channel"},
                )
                if isinstance(w, EmptyWidget):
                    raise ValueError("Can't create widget")
                self.channel_select_li.append(w)
                self.append(self.channel_select_li[-1])
        elif len(self.channel_select_li) > channels_num:
            for _ in range(channels_num, len(self.channel_select_li)):
                self.remove(self.channel_select_li[-1])
                self.channel_select_li.pop()

        self.change_channels_num(self.image())

    def change_channels_num(self, image):
        if image is None:
            self.image = lambda: None
            return
        self.image = ref(image)
        for el in self.channel_select_li:
            if hasattr(el, "change_channels_num"):
                el.change_channels_num(image.channels)

    @property
    def value(self) -> BioImageModel:
        return BioImageModel(
            self.file_select.value, [v.value for v in self.channel_select_li]
        )

    @value.setter
    def value(self, value: BioImageModel):
        self.file_select.value = value.path
        for ch, v in zip(self.channel_select_li, value.channels):
            ch.value = v


register_type(BioImageModel, widget_type=BioImageWidget)


def calc_number_of_channels(model_path):
    import bioimageio.core as bc
    from bioimageio.core.resource_io import nodes

    model = bc.load_resource_description(model_path)
    assert isinstance(model, nodes.Model)
    # TODO handle multiple inputs
    try:
        channel_index = model.inputs[0].axes.index("c")
    except ValueError:
        return 1
    if isinstance(model.inputs[0].shape, nodes.ParametrizedInputShape):
        return model.inputs[0].shape.min[channel_index]
    return model.inputs[0].shape[channel_index]
