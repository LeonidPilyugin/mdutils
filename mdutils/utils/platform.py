import enum


class Platform(enum.Enum):
    HIP = enum.auto()
    CUDA = enum.auto()

    @staticmethod
    def as_str(platform) -> str:
        match platform:
            case Platform.HIP:
                return "HIP"
            case Platform.CUDA:
                return "CUDA"

        assert not "Reached"

