from typing import NamedTuple, Tuple

from pathgen.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def make(cls, x, y, size, level):
        location = Point(x, y)
        size = Size(size, size)
        return Region(level, location, size)

    def as_values(self) -> Tuple[int, int, int, int, int]:
        return (
            self.location.x,
            self.location.y,
            self.size.width,
            self.size.height,
            self.level,
        )
