"""
Utilities for generating launcher icons for wrapped kernels.
"""

from __future__ import annotations

from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from jupyter_client.kernelspec import KernelSpec


KANGAROO_ASSET_SVG = "kangaroo.svg"


def _open_svg_resource(name: str, size: int) -> Optional[Image.Image]:
    """Rasterize a bundled SVG asset to a square RGBA image of the given size."""
    try:
        import cairosvg
    except ImportError:
        return None

    try:
        resource_path = resources.files(__package__).joinpath(f"data/{name}")
        svg_url = str(resource_path)
    except AttributeError:
        # Python 3.8 fallback
        with resources.open_binary(__package__, f"data/{name}") as fh:
            svg_bytes = fh.read()
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
        return Image.open(BytesIO(png_bytes)).convert("RGBA")

    png_bytes = cairosvg.svg2png(url=svg_url, output_width=size, output_height=size)
    return Image.open(BytesIO(png_bytes)).convert("RGBA")


def _rasterize_svg(svg_path: Path, size: int) -> Optional[Image.Image]:
    """Rasterize an SVG to a square RGBA image if cairosvg is available."""
    try:
        import cairosvg
    except ImportError:
        return None

    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=size,
        output_height=size,
    )
    return Image.open(BytesIO(png_bytes)).convert("RGBA")


def _find_icon_candidate(resource_dir: Path) -> Optional[Path]:
    """Return the first existing icon file in preferred order."""
    candidates: Sequence[str] = (
        "logo-64x64.png",
        "logo-64x64.svg",
        "logo-32x32.png",
        "logo-32x32.svg",
        "logo.png",
    )
    for filename in candidates:
        candidate = resource_dir / filename
        if candidate.exists():
            return candidate
    return None


def _load_base_icon(spec: KernelSpec) -> Optional[Image.Image]:
    """Load a base icon for a kernel spec, resized to 64x64."""
    resource_dir = Path(spec.resource_dir)
    icon_path = _find_icon_candidate(resource_dir)

    if icon_path is None:
        return None

    if icon_path.suffix.lower() == ".svg":
        image = _rasterize_svg(icon_path, size=64)
    else:
        with Image.open(icon_path) as img:
            image = img.copy()

    if image is None:
        return None

    return image.convert("RGBA").resize((64, 64), Image.LANCZOS)


def _overlay_kangaroo(base: Image.Image) -> Image.Image:
    """
    Place the kangaroo mark in the top-right corner without covering the base icon.

    Strategy: reserve a top-right strip for the badge; shrink and shift the base icon
    down/left so they don't overlap.
    """
    canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    # Badge sizing
    badge_size = 16
    badge_margin = 0  # keep it flush to the corner

    overlay = _open_svg_resource(KANGAROO_ASSET_SVG, size=badge_size)
    if overlay is None:  # fallback: no badge
        overlay = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))

    overlay.thumbnail((badge_size, badge_size), Image.LANCZOS)
    badge_pos = (
        canvas.width - overlay.width - badge_margin,
        badge_margin,
    )

    # Base icon: shrink to leave room for badge strip (height=badge_size)
    available_width = canvas.width - badge_size - badge_margin
    available_height = canvas.height - badge_size - badge_margin

    base_safe_width = min(available_width - 4, base.width)
    base_safe_height = min(available_height - 4, base.height)
    base_resized = base.copy().convert("RGBA").resize(
        (base_safe_width, base_safe_height),
        Image.LANCZOS,
    )
    base_pos = (
        (available_width - base_safe_width) // 2,
        badge_size + badge_margin + 2,  # push below the badge area
    )

    canvas.alpha_composite(base_resized, dest=base_pos)
    canvas.alpha_composite(overlay, dest=badge_pos)
    return canvas


def create_wrapped_kernel_icons(kernel_dir: Path, wrapped_spec: Optional[KernelSpec], logger=None) -> bool:
    """
    Create launcher icons for a wrapped kernel.

    Icons are saved as logo-64x64.png and logo-32x32.png inside kernel_dir.
    """
    try:
        base_icon = _load_base_icon(wrapped_spec) if wrapped_spec else None
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Failed to load base icon for wrapped kernel: %s", exc)
        base_icon = None

    if base_icon is None:
        base_icon = Image.new("RGBA", (64, 64), (245, 245, 245, 255))

    composed = _overlay_kangaroo(base_icon)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    try:
        (kernel_dir / "logo-64x64.png").parent.mkdir(parents=True, exist_ok=True)
        composed.save(kernel_dir / "logo-64x64.png")
        composed.resize((32, 32), Image.LANCZOS).save(kernel_dir / "logo-32x32.png")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Failed to write wrapped kernel icons: %s", exc)
        return False


def create_base_kernel_icons(kernel_dir: Path, logger=None) -> bool:
    """
    Create launcher icons for the base Jumper Wrapper Kernel itself.

    Uses the kangaroo SVG as the full icon (no badge overlay).
    """
    try:
        full_icon = _open_svg_resource(KANGAROO_ASSET_SVG, size=64)
        if full_icon is None:
            return False

        kernel_dir.mkdir(parents=True, exist_ok=True)
        full_icon.save(kernel_dir / "logo-64x64.png")
        full_icon.resize((32, 32), Image.LANCZOS).save(kernel_dir / "logo-32x32.png")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Failed to write base kernel icons: %s", exc)
        return False
