from dataclasses import dataclass
from rasterio.transform import from_origin
from rasterio.windows import Window
import numpy as np


@dataclass
class CubeGrid:
    name: str
    resolution: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: str = "EPSG:4326"
    chunks: tuple[int, int] = (512, 512)

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.resolution))

    @property
    def transform(self):
        return from_origin(
            self.xmin,
            self.ymax,
            self.resolution,
            self.resolution,
        )

    def x_coords(self):
        return self.xmin + self.resolution * (np.arange(self.width) + 0.5)

    def y_coords(self):
        return self.ymax - self.resolution * (np.arange(self.height) + 0.5)

    def iter_windows(self):
        chunk_y, chunk_x = self.chunks

        for row in range(0, self.height, chunk_y):
            h = min(chunk_y, self.height - row)

            for col in range(0, self.width, chunk_x):
                w = min(chunk_x, self.width - col)

                yield Window(
                    col_off=col,
                    row_off=row,
                    width=w,
                    height=h,
                )
