"""Render the profile animation, still poster, and self-hosted SVG badges."""

from __future__ import annotations

import json
import math
import os
import xml.sax.saxutils
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat, features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
OUT = HERE


def read_json(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


CONTENT = read_json("content.json")
TOKENS = read_json("tokens.json")
BRAIN = read_json("brain.json")
TRACE = read_json("trace.json")
COLORS = {key: tuple(bytes.fromhex(value.lstrip("#"))) for key, value in TOKENS["colors"].items()}
WIDTH = TOKENS["canvas"]["width"]
HEIGHT = TOKENS["canvas"]["height"]
SCALE = int(os.environ.get("PROMO_EXPORT", TOKENS["canvas"]["scale"]))
FRAME_RATE = int(os.environ.get("PROMO_Q", TOKENS["canvas"]["frame_rate"]))
FRAME_MS = TOKENS["timing"]["frame_ms"]
LAYOUT_FRAMES = (11, 33, 62, 104, 152, 172)
FRAME_COUNT = LAYOUT_FRAMES[-1] * FRAME_RATE
FRAME_DURATION = FRAME_MS // FRAME_RATE
IDENTITY = CONTENT["identity"]
UI = CONTENT["ui"]
TRACE_EPOCHS = TRACE["epochs"]
TRACE_GRID = TRACE["dataset"]["grid"]
LOOP_SECONDS = FRAME_COUNT * FRAME_DURATION / 1000
ACCENT_RGB = COLORS["ACCENT"]


@lru_cache(maxsize=64)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required vendored font is missing: {path}")
    loaded = ImageFont.truetype(str(path), size=size)
    if name == "SpaceGrotesk.ttf" and hasattr(loaded, "get_variation_axes"):
        axes = loaded.get_variation_axes()
        if axes:
            values = [axis["default"] for axis in axes]
            values[0] = min(500, axes[0]["maximum"])
            loaded.set_variation_by_axes(values)
    return loaded


def color(name: str) -> tuple[int, int, int]:
    return COLORS[name]


def text_width(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=face)


def check_glyphs() -> None:
    display = font(TOKENS["fonts"]["display"], 24)
    mono = font(TOKENS["fonts"]["mono"], 16)
    strings = [
        IDENTITY["name"],
        IDENTITY["handle"],
        IDENTITY["education"],
        IDENTITY["institution"],
        *IDENTITY["roles"],
        *IDENTITY["focus"],
        *CONTENT["skills"],
        *CONTENT["boot_lines"],
        *UI.values(),
        *(region["name"] for region in CONTENT["brain_regions"]),
        CONTENT["links"]["email"],
        CONTENT["links"]["github"],
    ]
    for value in strings:
        for character in value:
            if character.isspace():
                continue
            for face in (display, mono):
                try:
                    face.getmask(character)
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Font {face.path} cannot render {character!r}."
                    ) from error


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for item in rgb:
        value = item / 255
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    light, dark = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def check_contrast() -> None:
    required = TOKENS["accessibility"]
    bg = color("BG")
    if contrast(color("BODY"), bg) < required["body_contrast_min"]:
        raise ValueError("Body text contrast is below the configured WCAG threshold.")
    if contrast(color("FAINT"), bg) < required["label_contrast_min"]:
        raise ValueError("Label text contrast is below the configured WCAG threshold.")


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    face: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and text_width(draw, candidate, face) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    face: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 20,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, face, max_width):
        draw.text((x, y), line, font=face, fill=fill)
        y += line_gap
    return y


def frame_base(tick: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), color("BG"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, WIDTH - 16, HEIGHT - 16), outline=color("LINE"), width=1)
    draw.line((25, 32, 58, 32), fill=color("FAINT"), width=1)
    draw.line((25, 32, 25, 46), fill=color("FAINT"), width=1)
    draw.line((WIDTH - 25, HEIGHT - 32, WIDTH - 58, HEIGHT - 32), fill=color("FAINT"), width=1)
    draw.line((WIDTH - 25, HEIGHT - 32, WIDTH - 25, HEIGHT - 46), fill=color("FAINT"), width=1)
    for y in range(38, HEIGHT - 35, 4):
        draw.line((27, y, WIDTH - 27, y), fill=(12, 20, 23), width=1)
    # The ambient glyph field is deliberately static and faint to avoid flicker.
    glyphs = "0123456789ABCDEF"
    for column in range(9):
        x = 68 + column * 91
        offset = (column * 7) % 13
        for row in range(6):
            glyph = glyphs[(column * 5 + row * 3) % len(glyphs)]
            y = 55 + ((row * 47 + offset) % 292)
            draw.text((x, y), glyph, font=font(TOKENS["fonts"]["mono"], 10), fill=(22, 37, 40))
    draw.line((34, HEIGHT - 47, WIDTH - 34, HEIGHT - 47), fill=color("LINE"), width=1)
    draw.text((36, HEIGHT - 38), UI["status_label"].lower(), font=font(TOKENS["fonts"]["mono"], 10), fill=color("FAINT"))
    draw.text(
        (WIDTH - 255, HEIGHT - 39),
        "github.com/" + IDENTITY["handle"],
        font=font(TOKENS["fonts"]["mono"], 10),
        fill=color("FAINT"),
    )
    return image, draw


def draw_power(draw: ImageDraw.ImageDraw, tick: float) -> None:
    progress = min(1.0, max(0.0, tick / (LAYOUT_FRAMES[0] - 1)))
    half_height = max(1, int((HEIGHT * 0.37) * progress))
    y = HEIGHT // 2
    draw.line((92, y - half_height, WIDTH - 92, y - half_height), fill=color("BODY"), width=1)
    draw.line((92, y + half_height, WIDTH - 92, y + half_height), fill=color("BODY"), width=1)
    if progress > 0.55:
        draw.text((WIDTH // 2 - 67, y - 8), IDENTITY["handle"].lower(), font=font(TOKENS["fonts"]["mono"], 16), fill=color("MUTED"))


def draw_boot(draw: ImageDraw.ImageDraw, tick: float) -> None:
    draw.text((74, 66), UI["boot_heading"], font=font(TOKENS["fonts"]["mono"], 14), fill=color("MUTED"))
    visible = max(0, min(len(CONTENT["boot_lines"]), int((tick - 11) * 0.48)))
    for index, line in enumerate(CONTENT["boot_lines"][:visible]):
        y = 104 + index * 23
        draw.text((90, y), UI["boot_ok"], font=font(TOKENS["fonts"]["mono"], 13), fill=color("MUTED"))
        draw.text((150, y), line, font=font(TOKENS["fonts"]["mono"], 13), fill=color("BODY"))
    if tick >= 30:
        draw.text((74, 355), UI["terminal_prompt"], font=font(TOKENS["fonts"]["mono"], 18), fill=color("BODY"))


def role_text() -> str:
    return " | ".join(role.lower() for role in IDENTITY["roles"])


def draw_identity(draw: ImageDraw.ImageDraw, tick: float) -> None:
    draw.text((72, 69), f"{IDENTITY['handle'].lower()} {UI['prompt_suffix']}", font=font(TOKENS["fonts"]["mono"], 13), fill=color("MUTED"))
    draw.text((72, 93), UI["terminal_prompt"], font=font(TOKENS["fonts"]["mono"], 16), fill=color("BODY"))
    command_progress = max(0, min(len(UI["whoami"]), int((tick - 33) * 0.8)))
    draw.text((91, 96), UI["prompt_arrow"], font=font(TOKENS["fonts"]["mono"], 15), fill=color("ACCENT"))
    draw.text((108, 96), UI["whoami"][:command_progress], font=font(TOKENS["fonts"]["mono"], 15), fill=color("BODY"))
    if tick < 37:
        return
    x1, y1, x2, y2 = 88, 140, WIDTH - 88, 378
    draw.rounded_rectangle((x1, y1, x2, y2), radius=5, fill=color("RAISED"), outline=color("LINE"), width=1)
    title = IDENTITY["name"]
    draw.text((112, 157), title, font=font(TOKENS["fonts"]["display"], 27), fill=color("HEADING"))
    draw.line((112, 196, 252, 196), fill=color("ACCENT"), width=2)
    label_font = font(TOKENS["fonts"]["mono"], 11)
    body_font = font(TOKENS["fonts"]["mono"], 12)
    label_x, value_x, max_width = 112, 215, 540
    rows = [
        (UI["role_label"], role_text()),
        (UI["studying_label"], f"{IDENTITY['education'].lower()} | {IDENTITY['institution'].lower()}"),
        (UI["focus_label"], " | ".join(item.lower() for item in IDENTITY["focus"])),
    ]
    row_y = 215
    for index, (label, value) in enumerate(rows):
        draw.text((label_x, row_y), label + ":", font=label_font, fill=color("FAINT"))
        row_y = draw_wrapped(draw, (value_x, row_y), value, body_font, color("BODY"), max_width, 18)
        row_y += 8
        if row_y > y2 - 12 and index < len(rows) - 1:
            raise ValueError("Identity card content exceeds its layout.")
    if tick > 59:
        draw.text((WIDTH - 119, 157), "01 / 03", font=label_font, fill=color("FAINT"))


def draw_network(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], progress: float) -> None:
    x1, y1, x2, y2 = box
    layers = [2, 8, 8, 1]
    centers: list[list[tuple[int, int]]] = []
    for layer_index, count in enumerate(layers):
        x = x1 + int((x2 - x1) * layer_index / (len(layers) - 1))
        layer_centers = []
        for node_index in range(count):
            y = y1 + int((y2 - y1) * (node_index + 1) / (count + 1))
            layer_centers.append((x, y))
        centers.append(layer_centers)
    for left, right in zip(centers, centers[1:]):
        for start in left:
            for end in right:
                draw.line((start, end), fill=(29, 49, 53), width=1)
    lit_count = int(len(BRAIN["nodes"]) * progress)
    for layer_index, layer in enumerate(centers):
        for node_index, point in enumerate(layer):
            fill = color("MUTED") if (layer_index + node_index) < lit_count else color("FAINT")
            radius = 4 if layer_index in (0, 3) else 3
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)


def draw_training(draw: ImageDraw.ImageDraw, tick: float) -> None:
    start, end = LAYOUT_FRAMES[2], LAYOUT_FRAMES[3]
    progress = max(0.0, min(1.0, (tick - start) / (end - start)))
    shown = max(1, min(len(TRACE_EPOCHS), int(progress * len(TRACE_EPOCHS)) + 1))
    current = TRACE_EPOCHS[shown - 1]
    heading_font = font(TOKENS["fonts"]["mono"], 14)
    small_font = font(TOKENS["fonts"]["mono"], 10)
    draw.text((66, 62), "$ " + UI["training_command"], font=heading_font, fill=color("BODY"))
    draw.text((66, 84), UI["training_caption"], font=small_font, fill=color("MUTED"))
    draw_network(draw, (86, 145, 278, 318), progress)
    draw.text((98, 334), " - ".join(str(size) for size in TRACE["architecture"]), font=small_font, fill=color("FAINT"))

    plot = (324, 128, 584, 328)
    gx0, gy0, gx1, gy1 = plot
    grid_width, grid_height = TRACE_GRID["width"], TRACE_GRID["height"]
    grid = current["grid"]
    cell_w = (gx1 - gx0) / grid_width
    cell_h = (gy1 - gy0) / grid_height
    for row in range(grid_height):
        for column in range(grid_width):
            value = grid[row * grid_width + column]
            shade = int(18 + 22 * value)
            draw.rectangle(
                (
                    int(gx0 + column * cell_w),
                    int(gy0 + row * cell_h),
                    int(gx0 + (column + 1) * cell_w + 1),
                    int(gy0 + (row + 1) * cell_h + 1),
                ),
                fill=(shade, shade + 7, shade + 9),
            )
    for x, y, label in TRACE["dataset"]["points"]:
        px = gx0 + (x + 2.0) / 4.0 * (gx1 - gx0)
        py = gy0 + (1.8 - y) / 3.6 * (gy1 - gy0)
        radius = 2
        if label == 0:
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color("BODY"), width=1)
        else:
            draw.polygon(((px, py - radius), (px - radius, py + radius), (px + radius, py + radius)), fill=color("MUTED"))
    draw.rectangle(plot, outline=color("LINE"), width=1)

    chart = (632, 147, 807, 288)
    cx0, cy0, cx1, cy1 = chart
    draw.line((cx0, cy1, cx1, cy1), fill=color("LINE"), width=1)
    draw.line((cx0, cy0, cx0, cy1), fill=color("LINE"), width=1)
    losses = [float(epoch["loss"]) for epoch in TRACE_EPOCHS[:shown]]
    max_loss = max(0.01, float(TRACE_EPOCHS[0]["loss"]))
    points = [
        (
            cx0 + (cx1 - cx0) * index / max(1, len(losses) - 1),
            cy1 - (cy1 - cy0) * value / max_loss,
        )
        for index, value in enumerate(losses)
    ]
    if len(points) > 1:
        draw.line(points, fill=color("MUTED"), width=2)
        px, py = points[-1]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color("ACCENT"))
    draw.text((cx0, cy0 - 23), UI["loss_label"], font=small_font, fill=color("MUTED"))
    draw.text((632, 310), f"{UI['epoch_label']} {current['epoch']:>3}", font=small_font, fill=color("BODY"))
    draw.text((632, 330), f"{UI['accuracy_label']} {current['accuracy'] * 100:05.1f}%", font=small_font, fill=color("MUTED"))


def draw_brain(draw: ImageDraw.ImageDraw, tick: float) -> None:
    start, end = LAYOUT_FRAMES[3], LAYOUT_FRAMES[4]
    progress = max(0.0, min(1.0, (tick - start) / (end - start)))
    ox, oy, bw, bh = 72, 100, 350, 250
    points = [(ox + int(x * bw), oy + int(y * bh)) for x, y in BRAIN["outline"]]
    outline_progress = min(1.0, progress * 2.5)
    outline_color = tuple(
        int(color("BG")[channel] * (1 - outline_progress) + color("MUTED")[channel] * outline_progress)
        for channel in range(3)
    )
    draw.line(points + [points[0]], fill=outline_color, width=2, joint="curve")
    node_points = [(ox + int(x * bw), oy + int(y * bh)) for x, y in BRAIN["nodes"]]
    source_points: list[tuple[int, int]] = []
    layer_sizes = TRACE["architecture"]
    for layer_index, count in enumerate(layer_sizes):
        x = 86 + int((278 - 86) * layer_index / (len(layer_sizes) - 1))
        for node_index in range(count):
            y = 145 + int((318 - 145) * (node_index + 1) / (count + 1))
            source_points.append((x, y))

    edge_color = tuple(
        int((34, 58, 61)[channel] * min(1.0, progress * 2.0))
        for channel in range(3)
    )
    for index, (x, y) in enumerate(node_points):
        nearest = sorted(
            (
                (math.dist((x, y), other), other)
                for other_index, other in enumerate(node_points)
                if other_index != index and math.dist((x, y), other) < 46
            ),
            key=lambda item: item[0],
        )[:2]
        for _, point in nearest:
            if point[0] > x or (point[0] == x and point[1] > y):
                draw.line((x, y, point[0], point[1]), fill=edge_color, width=1)

    for index, target in enumerate(node_points[:BRAIN["trained_nodes"]]):
        source = source_points[index]
        travel = min(1.0, max(0.0, (progress - index * 0.012) / 0.45))
        x = round(source[0] * (1 - travel) + target[0] * travel)
        y = round(source[1] * (1 - travel) + target[1] * travel)
        fill = color("BODY") if travel >= 1.0 else color("MUTED")
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill)
    resting_progress = min(1.0, max(0.0, (progress - 0.45) * 2.0))
    for x, y in node_points[BRAIN["trained_nodes"]:]:
        fill = tuple(int(component * resting_progress) for component in color("FAINT"))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=fill)

    label_x, label_y = 488, 108
    mono = font(TOKENS["fonts"]["mono"], 12)
    region_slots: list[tuple[int, int]] = []
    for index, region in enumerate(CONTENT["brain_regions"]):
        y = label_y + index * 60
        region_slots.append((label_x + index * 36, y + 4))
        draw.text((label_x, y), region["name"], font=mono, fill=color("MUTED"))
        draw.text((label_x, y + 20), " / ".join(skill.lower() for skill in region["skills"]), font=font(TOKENS["fonts"]["mono"], 10), fill=color("BODY"))
    if region_slots:
        signal_index = min(len(region_slots) - 1, int(progress * len(region_slots)))
        signal_x, signal_y = region_slots[signal_index]
        draw.ellipse((signal_x - 3, signal_y - 3, signal_x + 3, signal_y + 3), fill=color("ACCENT"))
    draw.text((label_x, 356), UI["brain_label"], font=mono, fill=color("FAINT"))


def draw_signoff(draw: ImageDraw.ImageDraw, tick: float) -> None:
    last_tick = (FRAME_COUNT - 1) / FRAME_RATE
    progress = max(0.0, min(1.0, (tick - LAYOUT_FRAMES[4]) / (last_tick - LAYOUT_FRAMES[4])))
    fade = max(0.0, min(1.0, (1.0 - progress) * 3.0))
    def faded(name: str) -> tuple[int, int, int]:
        return tuple(int(component * fade) for component in color(name))

    cx, cy = 282, 211
    for index, (nx, ny) in enumerate(BRAIN["nodes"]):
        x, y = cx + int((nx - 0.5) * 150), cy + int((ny - 0.5) * 115)
        radius = 2 if index < BRAIN["trained_nodes"] else 1
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=faded("MUTED" if index < BRAIN["trained_nodes"] else "FAINT"))
    name = IDENTITY["name"]
    face = font(TOKENS["fonts"]["display"], 30)
    draw.text((410, 163), name, font=face, fill=faded("HEADING"))
    draw.text((412, 207), IDENTITY["handle"].lower(), font=font(TOKENS["fonts"]["mono"], 13), fill=faded("BODY"))
    draw.text((412, 239), CONTENT["links"]["github"].replace("https://", ""), font=font(TOKENS["fonts"]["mono"], 12), fill=faded("ACCENT"))
    draw.text((412, 266), CONTENT["links"]["email"], font=font(TOKENS["fonts"]["mono"], 12), fill=faded("BODY"))
    draw.text((412, 310), UI["signoff_label"], font=font(TOKENS["fonts"]["mono"], 10), fill=faded("FAINT"))


def draw_frame(tick: float) -> Image.Image:
    image, draw = frame_base(tick)
    if tick < LAYOUT_FRAMES[0]:
        draw_power(draw, tick)
    elif tick < LAYOUT_FRAMES[1]:
        draw_boot(draw, tick)
    elif tick < LAYOUT_FRAMES[2]:
        draw_identity(draw, tick)
    elif tick < LAYOUT_FRAMES[3]:
        draw_training(draw, tick)
    elif tick < LAYOUT_FRAMES[4]:
        draw_brain(draw, tick)
    else:
        draw_signoff(draw, tick)
    return image


def gif_palette() -> Image.Image:
    palette_colors: list[int] = []
    anchors = [color(key) for key in TOKENS["colors"]]
    for rgb in anchors:
        palette_colors.extend(rgb)
    start, end = color("BG"), color("HEADING")
    for index in range(56):
        amount = index / 55
        palette_colors.extend(
            int(start[channel] * (1 - amount) + end[channel] * amount)
            for channel in range(3)
        )
    while len(palette_colors) < 256 * 3:
        palette_colors.extend(color("BG"))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette_colors[: 256 * 3])
    return palette_image


@lru_cache(maxsize=1)
def shared_gif_palette() -> Image.Image:
    return gif_palette()


def high_resolution(image: Image.Image) -> Image.Image:
    return image.resize((WIDTH * SCALE, HEIGHT * SCALE), Image.Resampling.LANCZOS)


def frame_luminance(image: Image.Image) -> float:
    small = image.convert("L").resize((44, 22), Image.Resampling.BILINEAR)
    return ImageStat.Stat(small).mean[0] / 255


def check_frames(frames: list[Image.Image]) -> None:
    settings = TOKENS["accessibility"]
    luminances = [frame_luminance(frame) for frame in frames]
    threshold = settings["flash_luminance_delta"]
    window = max(1, round(1000 / FRAME_DURATION))
    for index in range(len(luminances)):
        start = max(1, index - window + 1)
        flashes = sum(
            abs(luminances[position] - luminances[position - 1]) > threshold
            for position in range(start, index + 1)
        )
        if flashes > settings["flashes_per_second"]:
            raise ValueError(f"Flash check failed near rendered frame {index}.")

    for index, frame in enumerate(frames):
        pixels = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        saturation = pixels.max(axis=2).astype(np.int16) - pixels.min(axis=2).astype(np.int16)
        saturated_fraction = float(
            np.count_nonzero((saturation > 150) & (pixels.max(axis=2) > 190))
            / (pixels.shape[0] * pixels.shape[1])
        )
        if saturated_fraction > settings["accent_pixel_fraction_max"]:
            raise ValueError(f"Saturated-color budget exceeded in frame {index}.")


def render_frames(full_color: bool) -> list[Image.Image]:
    result: list[Image.Image] = []
    for index in range(FRAME_COUNT):
        tick = index / FRAME_RATE
        base = draw_frame(tick)
        if full_color:
            result.append(high_resolution(base))
        else:
            result.append(high_resolution(base).quantize(palette=shared_gif_palette(), dither=Image.Dither.NONE))
    return result


def save_poster() -> None:
    poster = high_resolution(draw_frame(48))
    poster.save(OUT / "lordjohan-terminal-poster.png", optimize=True)


def svg_badges() -> None:
    github_label = xml.sax.saxutils.escape(CONTENT["badge_labels"]["github"])
    email_label = xml.sax.saxutils.escape(CONTENT["badge_labels"]["email"])
    github_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="116" height="20" viewBox="0 0 116 20" role="img" aria-label="GitHub profile">
<rect width="116" height="20" rx="2" fill="{TOKENS['colors']['RAISED']}"/>
<path d="M8 10a4 4 0 1 0 8 0 4 4 0 0 0-8 0m2-1h4m-3-2v6" fill="none" stroke="{TOKENS['colors']['ACCENT']}" stroke-width="1.2"/>
<text x="23" y="14" fill="{TOKENS['colors']['BODY']}" font-family="JetBrains Mono,monospace" font-size="10">{github_label}</text>
</svg>
"""
    email_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="94" height="20" viewBox="0 0 94 20" role="img" aria-label="Email Aria Raoufi">
<rect width="94" height="20" rx="2" fill="{TOKENS['colors']['RAISED']}"/>
<path d="M8 6h12v9H8zM8 7l6 4 6-4" fill="none" stroke="{TOKENS['colors']['ACCENT']}" stroke-width="1.2"/>
<text x="25" y="14" fill="{TOKENS['colors']['BODY']}" font-family="JetBrains Mono,monospace" font-size="10">{email_label}</text>
</svg>
"""
    badge_dir = ASSETS / "badges"
    badge_dir.mkdir(parents=True, exist_ok=True)
    (badge_dir / "github.svg").write_text(github_svg, encoding="utf-8")
    (badge_dir / "email.svg").write_text(email_svg, encoding="utf-8")
    for name in ("github.svg", "email.svg"):
        if (badge_dir / name).stat().st_size >= 2048:
            raise ValueError(f"Badge exceeds its 2 KB budget: {name}")


def validate_trace() -> None:
    if TRACE["architecture"] != [2, 8, 8, 1]:
        raise ValueError("Training trace architecture does not match 2-8-8-1.")
    if TRACE["final_accuracy"] < 0.95:
        raise ValueError("Training trace accuracy is below 95%.")
    expected_grid = TRACE_GRID["width"] * TRACE_GRID["height"]
    if any(len(epoch["grid"]) != expected_grid for epoch in TRACE_EPOCHS):
        raise ValueError("A training checkpoint has an invalid decision-grid size.")


def main() -> None:
    if SCALE < 1 or FRAME_RATE < 1:
        raise ValueError("PROMO_EXPORT and PROMO_Q must be positive integers.")
    if FRAME_MS % FRAME_RATE or FRAME_DURATION % 10 or FRAME_DURATION < 10:
        raise ValueError("Frame timings must be GIF-compatible 10 ms multiples.")
    if LOOP_SECONDS > TOKENS["timing"]["loop_limit_seconds"]:
        raise ValueError(f"Animation loop is too long: {LOOP_SECONDS:.2f}s.")
    if not features.check("webp"):
        raise RuntimeError("Pillow was built without animated WebP support.")
    validate_trace()
    check_contrast()
    check_glyphs()
    svg_badges()
    save_poster()

    gif_frames = render_frames(full_color=False)
    check_frames(gif_frames)
    gif_frames[0].save(
        OUT / "lordjohan-terminal.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=FRAME_DURATION,
        loop=0,
        optimize=True,
        disposal=1,
    )
    del gif_frames

    webp_frames = render_frames(full_color=True)
    webp_frames[0].save(
        OUT / "lordjohan-terminal.webp",
        format="WEBP",
        save_all=True,
        append_images=webp_frames[1:],
        duration=FRAME_DURATION,
        loop=0,
        quality=78,
        method=6,
    )
    del webp_frames

    for name, limit in (("lordjohan-terminal.gif", 5_000_000), ("lordjohan-terminal.webp", 3_000_000)):
        size = (OUT / name).stat().st_size
        if size > limit:
            raise ValueError(f"{name} exceeds its {limit:,}-byte budget ({size:,} bytes).")
    print(
        f"Rendered {FRAME_COUNT} frames ({LOOP_SECONDS:.2f}s): "
        f"GIF {(OUT / 'lordjohan-terminal.gif').stat().st_size:,} bytes, "
        f"WebP {(OUT / 'lordjohan-terminal.webp').stat().st_size:,} bytes."
    )


if __name__ == "__main__":
    main()
