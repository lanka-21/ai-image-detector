from torchvision.datasets import ImageFolder

class RobustImageFolder(ImageFolder):
    def __getitem__(self, index):
        for _ in range(3):
            try:
                return super().__getitem__(index)
            except Exception:
                index = (index + 1) % len(self)

        raise RuntimeError("Too many corrupted images")