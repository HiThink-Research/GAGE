"""Image rendering helpers for Gomoku arena observations."""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme

_COLOR_BACKGROUND = (240, 211, 154)
_COLOR_BOARD = (216, 168, 91)
_COLOR_GRID = (94, 64, 34)
_COLOR_LABEL = (77, 50, 22)
_COLOR_STAR = (63, 40, 18)
_COLOR_BLACK = (26, 26, 26)
_COLOR_BLACK_HIGHLIGHT = (102, 102, 102)
_COLOR_WHITE = (248, 246, 241)
_COLOR_WHITE_HIGHLIGHT = (255, 255, 255)
_COLOR_WHITE_OUTLINE = (183, 177, 168)
_COLOR_WIN = (230, 180, 34)
_COLOR_LAST_MOVE = (214, 69, 65)
_COLOR_LAST_MOVE_OUTLINE = (246, 211, 101)

_BITMAP_FONT = {
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"),
    "2": (" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"),
    "3": (" ### ", "#   #", "    #", " ### ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#    ", "#### ", "    #", "#   #", " ### "),
    "6": (" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "B": ("#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "D": ("#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "F": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "),
    "G": (" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "I": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"),
    "J": ("#####", "   # ", "   # ", "   # ", "#  # ", "#  # ", " ##  "),
    "K": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
}


@dataclass
class _RgbCanvas:
    """A tiny RGB canvas with basic raster drawing helpers."""

    width: int
    height: int
    background: tuple[int, int, int]

    def __post_init__(self) -> None:
        self.pixels = bytearray(bytes(self.background) * (self.width * self.height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Write one pixel when the target coordinate is within bounds."""

        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        index = ((y * self.width) + x) * 3
        self.pixels[index : index + 3] = bytes(color)

    def fill_rect(
        self,
        *,
        left: int,
        top: int,
        right: int,
        bottom: int,
        color: tuple[int, int, int],
    ) -> None:
        """Fill one axis-aligned rectangle."""

        clipped_left = max(0, int(left))
        clipped_top = max(0, int(top))
        clipped_right = min(self.width, int(right))
        clipped_bottom = min(self.height, int(bottom))
        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            return
        row_bytes = bytes(color) * (clipped_right - clipped_left)
        for y in range(clipped_top, clipped_bottom):
            index = ((y * self.width) + clipped_left) * 3
            self.pixels[index : index + len(row_bytes)] = row_bytes

    def draw_line(
        self,
        *,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw one straight line using Bresenham rasterization."""

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        radius = max(0, int(thickness) // 2)

        while True:
            self.fill_rect(
                left=x0 - radius,
                top=y0 - radius,
                right=x0 + radius + 1,
                bottom=y0 + radius + 1,
                color=color,
            )
            if x0 == x1 and y0 == y1:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy

    def fill_circle(
        self,
        *,
        center_x: int,
        center_y: int,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        """Fill one circle."""

        squared_radius = radius * radius
        for y in range(center_y - radius, center_y + radius + 1):
            delta_y = y - center_y
            for x in range(center_x - radius, center_x + radius + 1):
                delta_x = x - center_x
                if (delta_x * delta_x) + (delta_y * delta_y) <= squared_radius:
                    self.set_pixel(x, y, color)

    def draw_circle(
        self,
        *,
        center_x: int,
        center_y: int,
        radius: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw one circle outline."""

        outer = radius * radius
        inner_radius = max(0, radius - max(1, int(thickness)))
        inner = inner_radius * inner_radius
        for y in range(center_y - radius, center_y + radius + 1):
            delta_y = y - center_y
            for x in range(center_x - radius, center_x + radius + 1):
                delta_x = x - center_x
                dist = (delta_x * delta_x) + (delta_y * delta_y)
                if inner <= dist <= outer:
                    self.set_pixel(x, y, color)

    def draw_text(
        self,
        *,
        left: int,
        top: int,
        text: str,
        scale: int,
        color: tuple[int, int, int],
    ) -> None:
        """Render one string with a tiny built-in 5x7 bitmap font."""

        cursor_x = int(left)
        char_scale = max(1, int(scale))
        for char in str(text):
            glyph = _BITMAP_FONT.get(char.upper(), _BITMAP_FONT[" "])
            for row, pattern in enumerate(glyph):
                for col, pixel in enumerate(pattern):
                    if pixel != "#":
                        continue
                    self.fill_rect(
                        left=cursor_x + (col * char_scale),
                        top=int(top) + (row * char_scale),
                        right=cursor_x + ((col + 1) * char_scale),
                        bottom=int(top) + ((row + 1) * char_scale),
                        color=color,
                    )
            cursor_x += (len(glyph[0]) * char_scale) + char_scale

    def to_png_bytes(self) -> bytes:
        """Encode the canvas as a PNG image."""

        # STEP 1: Build scanlines using PNG filter type 0.
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            start = y * stride
            rows.append(b"\x00" + bytes(self.pixels[start : start + stride]))
        compressed = zlib.compress(b"".join(rows), level=9)

        # STEP 2: Assemble the PNG chunk stream.
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
        )
        idat = _png_chunk(b"IDAT", compressed)
        iend = _png_chunk(b"IEND", b"")
        return header + ihdr + idat + iend


def build_gomoku_observation_image_payload(
    *,
    board: Sequence[Sequence[str]],
    coord_scheme: str,
    last_move: Optional[str] = None,
    winning_line: Optional[Sequence[str]] = None,
    cell_px: int = 36,
    margin_px: int = 40,
) -> Optional[dict[str, Any]]:
    """Render one Gomoku board image payload for multimodal prompts.

    Args:
        board: Board tokens indexed by ``board[row][col]`` from bottom to top.
        coord_scheme: Coordinate scheme used by the environment.
        last_move: Optional last-move coordinate to highlight.
        winning_line: Optional winning-line coordinates to highlight.
        cell_px: Pixel spacing between adjacent intersections.
        margin_px: Outer margin used for coordinate labels.

    Returns:
        An image payload containing a PNG data URL, or ``None`` when rendering
        cannot be completed.
    """

    board_rows = [list(row) for row in board]
    board_size = len(board_rows)
    if board_size < 1 or any(len(row) != board_size for row in board_rows):
        return None

    coord_codec = GomokuCoordCodec(
        board_size=board_size,
        coord_scheme=normalize_coord_scheme(coord_scheme),
    )
    cell_size = max(24, int(cell_px))
    margin = max(28, int(margin_px))
    board_span = cell_size * max(0, board_size - 1)
    canvas = _RgbCanvas(
        width=board_span + (margin * 2),
        height=board_span + (margin * 2),
        background=_COLOR_BACKGROUND,
    )
    text_scale = max(1, cell_size // 14)

    # STEP 1: Draw the board plane, coordinate labels, grid lines, and star points.
    _draw_board_surface(
        canvas=canvas,
        board_size=board_size,
        coord_codec=coord_codec,
        cell_size=cell_size,
        margin=margin,
        text_scale=text_scale,
    )

    winning_line_set = {str(item) for item in (winning_line or []) if item}
    normalized_last_move = str(last_move) if last_move else None

    # STEP 2: Draw stones and move highlights on the occupied intersections.
    for row in range(board_size):
        for col in range(board_size):
            token = str(board_rows[row][col])
            if token == ".":
                continue
            coord = coord_codec.index_to_coord(row, col)
            center_x, center_y = _board_xy(
                board_size=board_size,
                row=row,
                col=col,
                cell_size=cell_size,
                margin=margin,
            )
            _draw_stone(
                canvas=canvas,
                center_x=center_x,
                center_y=center_y,
                cell_size=cell_size,
                token=token,
                is_last_move=coord == normalized_last_move,
                is_winning_move=coord in winning_line_set,
            )

    image_bytes = canvas.to_png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return {
        "data_url": data_url,
        "width": canvas.width,
        "height": canvas.height,
        "format": "png",
    }


def _draw_board_surface(
    *,
    canvas: _RgbCanvas,
    board_size: int,
    coord_codec: GomokuCoordCodec,
    cell_size: int,
    margin: int,
    text_scale: int,
) -> None:
    """Draw labels and grid lines for a Gomoku board."""

    left = margin
    top = margin
    right = margin + (cell_size * max(0, board_size - 1))
    bottom = margin + (cell_size * max(0, board_size - 1))

    canvas.fill_rect(
        left=left - 18,
        top=top - 18,
        right=right + 19,
        bottom=bottom + 19,
        color=_COLOR_BOARD,
    )
    canvas.draw_line(x0=left - 18, y0=top - 18, x1=right + 18, y1=top - 18, color=_COLOR_GRID, thickness=3)
    canvas.draw_line(x0=left - 18, y0=bottom + 18, x1=right + 18, y1=bottom + 18, color=_COLOR_GRID, thickness=3)
    canvas.draw_line(x0=left - 18, y0=top - 18, x1=left - 18, y1=bottom + 18, color=_COLOR_GRID, thickness=3)
    canvas.draw_line(x0=right + 18, y0=top - 18, x1=right + 18, y1=bottom + 18, color=_COLOR_GRID, thickness=3)

    for idx in range(board_size):
        offset = idx * cell_size
        canvas.draw_line(
            x0=left + offset,
            y0=top,
            x1=left + offset,
            y1=bottom,
            color=_COLOR_GRID,
            thickness=2,
        )
        canvas.draw_line(
            x0=left,
            y0=top + offset,
            x1=right,
            y1=top + offset,
            color=_COLOR_GRID,
            thickness=2,
        )

    for col, label in enumerate(coord_codec.column_labels()):
        center_x, _ = _board_xy(
            board_size=board_size,
            row=0,
            col=col,
            cell_size=cell_size,
            margin=margin,
        )
        _draw_centered_text(
            canvas=canvas,
            center_x=center_x,
            center_y=top - 22,
            text=str(label),
            scale=text_scale,
            color=_COLOR_LABEL,
        )

    for row in range(board_size):
        _, center_y = _board_xy(
            board_size=board_size,
            row=row,
            col=0,
            cell_size=cell_size,
            margin=margin,
        )
        _draw_centered_text(
            canvas=canvas,
            center_x=left - 24,
            center_y=center_y,
            text=str(row + 1),
            scale=text_scale,
            color=_COLOR_LABEL,
        )

    for row, col in _star_point_positions(board_size):
        center_x, center_y = _board_xy(
            board_size=board_size,
            row=row,
            col=col,
            cell_size=cell_size,
            margin=margin,
        )
        canvas.fill_circle(
            center_x=center_x,
            center_y=center_y,
            radius=max(2, cell_size // 10),
            color=_COLOR_STAR,
        )


def _draw_stone(
    *,
    canvas: _RgbCanvas,
    center_x: int,
    center_y: int,
    cell_size: int,
    token: str,
    is_last_move: bool,
    is_winning_move: bool,
) -> None:
    """Draw one stone plus optional last-move and winning-line highlights."""

    radius = max(9, int(cell_size * 0.38))
    stone_fill, stone_outline, stone_highlight = _stone_palette(token)
    canvas.fill_circle(center_x=center_x, center_y=center_y, radius=radius, color=stone_fill)
    canvas.draw_circle(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        color=stone_outline,
        thickness=2,
    )
    canvas.fill_circle(
        center_x=center_x - max(2, radius // 3),
        center_y=center_y - max(2, radius // 3),
        radius=max(2, radius // 5),
        color=stone_highlight,
    )
    if is_winning_move:
        canvas.draw_circle(
            center_x=center_x,
            center_y=center_y,
            radius=radius + 4,
            color=_COLOR_WIN,
            thickness=3,
        )
    if is_last_move:
        canvas.fill_circle(
            center_x=center_x,
            center_y=center_y,
            radius=max(3, radius // 5),
            color=_COLOR_LAST_MOVE,
        )
        canvas.draw_circle(
            center_x=center_x,
            center_y=center_y,
            radius=max(3, radius // 5),
            color=_COLOR_LAST_MOVE_OUTLINE,
            thickness=1,
        )


def _board_xy(
    *,
    board_size: int,
    row: int,
    col: int,
    cell_size: int,
    margin: int,
) -> tuple[int, int]:
    """Map one board index pair to image-space coordinates."""

    x = margin + (col * cell_size)
    y = margin + ((board_size - 1 - row) * cell_size)
    return x, y


def _draw_centered_text(
    *,
    canvas: _RgbCanvas,
    center_x: int,
    center_y: int,
    text: str,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    """Draw one string centered around a target point."""

    width = _text_width(text, scale)
    height = 7 * max(1, int(scale))
    canvas.draw_text(
        left=int(center_x - (width / 2)),
        top=int(center_y - (height / 2)),
        text=text,
        scale=scale,
        color=color,
    )


def _text_width(text: str, scale: int) -> int:
    """Return the raster width of one bitmap string."""

    char_scale = max(1, int(scale))
    if not text:
        return 0
    return (len(text) * 5 * char_scale) + ((len(text) - 1) * char_scale)


def _star_point_positions(board_size: int) -> list[tuple[int, int]]:
    """Return standard star-point coordinates for common Gomoku board sizes."""

    if board_size >= 13:
        anchors = [3, board_size // 2, board_size - 4]
    elif board_size >= 9:
        anchors = [2, board_size // 2, board_size - 3]
    elif board_size >= 5:
        anchors = [board_size // 2]
    else:
        return []
    return [(row, col) for row in anchors for col in anchors]


def _stone_palette(token: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return fill and outline colors for one board token."""

    normalized = str(token or "").strip().upper()
    if normalized.startswith(("W", "O")):
        return _COLOR_WHITE, _COLOR_WHITE_OUTLINE, _COLOR_WHITE_HIGHLIGHT
    return _COLOR_BLACK, (0, 0, 0), _COLOR_BLACK_HIGHLIGHT


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk with length and CRC fields."""

    checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", checksum)
    )
