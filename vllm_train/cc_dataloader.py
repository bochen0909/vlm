from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pyarrow.parquet as pq
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


@dataclass(frozen=True)
class CCExample:
    image_path: Path
    caption: str


class CCImageCaptionDataset(Dataset):
    """
    Torch-style Dataset for Conceptual Captions images downloaded via img2dataset.

    Returns by default: (PIL.Image, caption)
    Set `return_image_path=True` to return (Path, caption) instead.
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        return_image_path: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        self.images_root = Path(dataset_root, "cc_images")
        self.index_parquet = Path(dataset_root, "conceptual-captions-200k.parquet")

        self.return_image_path = return_image_path
        self.transform = transform
        self._examples: list[CCExample] = self._build_index()

    def _build_image_paths(self):
        # Loop recursively 2 directories down and find all jpg files
        jpg_files = {}
        for subdir1 in self.images_root.iterdir():
            if not subdir1.is_dir():
                continue
            for file in subdir1.iterdir():
                if file.is_file() and file.suffix.lower() == ".jpg":
                    file_idx = int(file.name.split(".")[0])
                    jpg_files[file_idx] = os.path.join(
                        self.images_root, subdir1.name, file.name
                    )
        return jpg_files

    def _load_caption_index(self) -> Dict[str, str]:
        table = pq.read_table(self.index_parquet, columns=["url", "caption"])
        urls = table["url"].to_pylist()
        caps = table["caption"].to_pylist()

        url_to_caption: Dict[str, str] = {}
        for u, c in zip(urls, caps):
            if u is None:
                continue
            if c is None:
                continue
            url_to_caption[str(u)] = str(c)
        return url_to_caption

    def _build_index(self) -> list[CCExample]:
        image_files = self._build_image_paths()
        url_to_caption = self._load_caption_index()

        table = pq.read_table(self.index_parquet, columns=["url", "caption"])
        captions = table["caption"].to_pylist()
        out: list[CCExample] = []
        for idx, caption in enumerate(captions):
            if idx in image_files:
                out.append(
                    CCExample(
                        image_path=image_files[idx],
                        caption=caption,
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any] | Dict[str, Any]:
        ex = self._examples[idx]
        caption: Any = ex.caption

        # Handle return_image_path flag
        if self.return_image_path:
            image_or_path: Any = Path(ex.image_path)
        else:
            with Image.open(ex.image_path) as im:
                image_or_path = im.convert("RGB").copy()

            # Apply image transform if specified
            # NOTE: For DataLoader batching, transform should convert PIL to tensor
            # Use transforms.ToTensor() or a compose that includes it
            if self.transform is not None:
                image_or_path = self.transform(image_or_path)

        return image_or_path, caption


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = CCImageCaptionDataset(transform=transform)
    example = dataset[16200]
    print(
        f"Single example - image shape: {example[0].shape}, caption: {example[1][:50]}..."
    )

    # Create DataLoader - images will be batched into tensors
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for batch in dataloader:
        images, captions = batch
        print(f"Batch - images shape: {images.shape}")  # Should be [16, C, H, W]
        print(f"Batch - captions: {len(captions)} items")
        print(f"First caption: {captions[0]}")
        break
