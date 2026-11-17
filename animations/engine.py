"""Tiny animation engine: matplotlib frames piped to ffmpeg (H.264), in a dark blackboard style.

A *scene* is a function ``draw(fig, s)`` that paints one frame for the time ``s`` (seconds since the
scene started).  ``Movie.play(scene, seconds)`` renders it; scenes are concatenated into one file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle   # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 24
BG, FG, MUTED, DIM = "#0e1116", "#ece6e2", "#9aa1ab", "#3a404a"
BLUE, GREEN, YELLOW, RED = "#58C4DD", "#83C167", "#F4D345", "#FC6255"
TEAL, PURPLE, ORANGE, PANEL = "#5CD0B3", "#B189F8", "#FF862F", "#161b22"

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans", "text.color": FG,
                     "axes.edgecolor": DIM, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


# ----------------------------------------------------------------------------- timing helpers
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ramp(s, start, end):
    """0 before ``start``, 1 after ``end``, eased in between."""
    return float(smooth((s - start) / max(end - start, 1e-9)))


def pulse(s, start, end):
    """Rises to 1 in the middle of [start, end] and returns to 0."""
    x = np.clip((s - start) / max(end - start, 1e-9), 0, 1)
    return float(np.sin(np.pi * x))


# ----------------------------------------------------------------------------- movie writer
class Movie:
    def __init__(self, path, crf: int = 21):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)
        self.frames = 0

    def play(self, scene, seconds: float) -> None:
        count = int(round(seconds * FPS))
        for k in range(count):
            self.fig.clf()
            self.fig.patch.set_facecolor(BG)
            scene(self.fig, k / FPS)
            self.fig.canvas.draw()
            frame = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frames += 1

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()
        plt.close(self.fig)


def still(scene, s: float, path) -> None:
    """Render one frame of a scene to a PNG (for checking layouts)."""
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    scene(fig, s)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------------------------- drawing helpers
def canvas(fig):
    """Full-frame axes in 'screen' units: x in [0, 16], y in [0, 9]."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(BG)
    return ax


def heading(ax, text, sub=None, alpha=1.0):
    ax.text(0.6, 8.35, text, fontsize=25, fontweight="bold", color=FG, alpha=alpha, va="center")
    if sub:
        import textwrap
        lines = textwrap.wrap(sub, 125)
        size = 13.5 if len(lines) == 1 else 12
        for k, line in enumerate(lines[:2]):
            ax.text(0.62, 7.9 - 0.36 * k, line, fontsize=size, color=MUTED, alpha=alpha, va="center")


def label(ax, x, y, text, color=FG, size=14, alpha=1.0, ha="left", va="center", weight="normal", **kw):
    if alpha <= 0.01:
        return
    ax.text(x, y, text, color=color, fontsize=size, alpha=min(alpha, 1.0), ha=ha, va=va, fontweight=weight, **kw)


def box(ax, x, y, w, h, text="", color=BLUE, alpha=1.0, size=13, glow=0.0, text_color=FG):
    """Rounded box centred at (x, y)."""
    if alpha <= 0.01:
        return
    face = matplotlib.colors.to_rgba(color, (0.16 + 0.4 * glow) * alpha)
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                           facecolor=face, edgecolor=matplotlib.colors.to_rgba(color, alpha), linewidth=1.6 + 1.5 * glow)
    ax.add_patch(patch)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=text_color, alpha=alpha, linespacing=1.35)


def arrow(ax, p0, p1, color=MUTED, alpha=1.0, width=1.6, frac=1.0):
    if alpha <= 0.01 or frac <= 0.02:
        return
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    end = p0 + frac * (p1 - p0)
    ax.annotate("", xy=end, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=width, alpha=alpha,
                                                       shrinkA=0, shrinkB=0, mutation_scale=14))


def partial(xy, frac):
    """The first ``frac`` (by arc length) of a polyline."""
    xy = np.asarray(xy, float)
    if frac >= 1.0:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = max(frac, 0.0) * cum[-1]
    k = int(np.searchsorted(cum, target, side="right") - 1)
    k = min(k, len(seg) - 1)
    tail = xy[k] + (target - cum[k]) / max(seg[k], 1e-9) * (xy[k + 1] - xy[k])
    return np.vstack([xy[: k + 1], tail])


def draw_path(ax, xy, frac=1.0, color=FG, width=2.5, alpha=1.0, style="-", zorder=3, head=False):
    if frac <= 0.0 or alpha <= 0.01:
        return None
    part = partial(xy, frac)
    ax.plot(part[:, 0], part[:, 1], color=color, lw=width, alpha=alpha, linestyle=style, zorder=zorder,
            solid_capstyle="round", dash_capstyle="round")
    if head:
        ax.add_patch(Circle(part[-1], 0.09, color=color, alpha=alpha, zorder=zorder + 1))
    return part[-1]


def tube_polygon(centre, radius):
    """Outline of the union of discs of the given radii along a curve (offset curves plus round caps)."""
    centre, radius = np.asarray(centre, float), np.asarray(radius, float)
    tangent = np.gradient(centre, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left, right = centre + radius[:, None] * normal, centre - radius[:, None] * normal
    angle_end = np.arctan2(tangent[-1, 1], tangent[-1, 0])
    arc = np.linspace(np.pi / 2, -np.pi / 2, 16)
    cap_end = centre[-1] + radius[-1] * np.column_stack([np.cos(angle_end + arc), np.sin(angle_end + arc)])
    angle_start = np.arctan2(tangent[0, 1], tangent[0, 0])
    arc = np.linspace(-np.pi / 2, -3 * np.pi / 2, 16)
    cap_start = centre[0] + radius[0] * np.column_stack([np.cos(angle_start + arc), np.sin(angle_start + arc)])
    return np.vstack([left, cap_end, right[::-1], cap_start])


def draw_tube(ax, centre, radius, frac=1.0, color=TEAL, alpha=0.22, edge_alpha=0.7, zorder=1):
    count = max(3, int(round(frac * len(centre))))
    if frac <= 0.02:
        return
    polygon = tube_polygon(centre[:count], radius[:count])
    ax.add_patch(Polygon(polygon, closed=True, facecolor=color, alpha=alpha, edgecolor="none", zorder=zorder))
    ax.add_patch(Polygon(polygon, closed=True, facecolor="none", edgecolor=color, alpha=edge_alpha, lw=1.2, zorder=zorder))


def stick_figure(ax, x, y, scale=1.0, color=FG, alpha=1.0, wave=0.0):
    """A person standing at (x, y) (feet).  ``wave`` in [0, 1] raises the right arm (throwing)."""
    s = scale
    kw = dict(color=color, lw=3.0, alpha=alpha, solid_capstyle="round")
    ax.add_patch(Circle((x, y + 2.05 * s), 0.28 * s, facecolor="none", edgecolor=color, lw=3.0, alpha=alpha))
    ax.plot([x, x], [y + 1.77 * s, y + 0.85 * s], **kw)                         # torso
    ax.plot([x, x - 0.35 * s], [y + 0.85 * s, y], **kw)                         # legs
    ax.plot([x, x + 0.35 * s], [y + 0.85 * s, y], **kw)
    ax.plot([x, x - 0.45 * s], [y + 1.5 * s, y + 1.05 * s], **kw)               # left arm
    hand = (x + 0.5 * s, y + (1.1 + 0.9 * wave) * s)
    ax.plot([x, hand[0]], [y + 1.5 * s, hand[1]], **kw)                         # right arm
    return hand


def drone_icon(ax, x, y, size=0.16, color=BLUE, alpha=1.0, zorder=6):
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ax.plot([x, x + dx * size], [y, y + dy * size], color=color, lw=1.6, alpha=alpha, zorder=zorder)
        ax.add_patch(Circle((x + dx * size, y + dy * size), size * 0.55, facecolor="none", edgecolor=color,
                            lw=1.3, alpha=alpha, zorder=zorder))
    ax.add_patch(Circle((x, y), size * 0.35, color=color, alpha=alpha, zorder=zorder))


def car_icon(ax, x, y, theta, size=0.3, color=BLUE, alpha=1.0, zorder=6):
    body = np.array([[-1.0, -0.55], [1.0, -0.55], [1.25, 0.0], [1.0, 0.55], [-1.0, 0.55]]) * size
    c, s = np.cos(theta), np.sin(theta)
    rot = body @ np.array([[c, s], [-s, c]])
    ax.add_patch(Polygon(rot + [x, y], closed=True, facecolor=matplotlib.colors.to_rgba(color, 0.35),
                         edgecolor=color, lw=1.8, alpha=alpha, zorder=zorder))


class WorldMap:
    """Maps world coordinates [m] into a rectangle of the screen, with equal scaling."""

    def __init__(self, rect, bounds, margin=0.25):
        x0, y0, x1, y1 = rect
        (wx0, wx1), (wy0, wy1) = bounds
        wx0, wx1, wy0, wy1 = wx0 - margin, wx1 + margin, wy0 - margin, wy1 + margin
        self.scale = min((x1 - x0) / (wx1 - wx0), (y1 - y0) / (wy1 - wy0))
        self.origin = np.array([(x0 + x1) / 2 - self.scale * (wx0 + wx1) / 2, (y0 + y1) / 2 - self.scale * (wy0 + wy1) / 2])
        self.rect = rect

    def pt(self, xy):
        return self.origin + self.scale * np.asarray(xy, float)[..., :2]

    def frame(self, ax, alpha=1.0, title=None):
        x0, y0, x1, y1 = self.rect
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.0,rounding_size=0.15",
                                    facecolor=PANEL, edgecolor=DIM, lw=1.2, alpha=alpha, zorder=0))
        if title:
            label(ax, x0 + 0.15, y1 - 0.25, title, MUTED, 11, alpha)

    def obstacles(self, ax, obstacles, alpha=1.0, labels=True, size=10):
        tint = {"person": RED, "chair": PURPLE, "stop_sign": ORANGE}
        for obstacle in obstacles:
            lo, hi = self.pt(obstacle["lo"]), self.pt(obstacle["hi"])
            colour = tint.get(obstacle["label"], MUTED)
            ax.add_patch(Rectangle(lo, *(hi - lo), facecolor=matplotlib.colors.to_rgba(colour, 0.28), edgecolor=colour,
                                   lw=1.4, alpha=alpha, zorder=2))
            if labels:
                label(ax, (lo[0] + hi[0]) / 2, hi[1] + 0.17, obstacle["label"].replace("_", " "), colour, size, alpha, ha="center")

    def endpoints(self, ax, start, goal, alpha=1.0, goal_alpha=None, size=10):
        s, g = self.pt(start), self.pt(goal)
        ax.add_patch(Circle(s, 0.1, color=FG, alpha=alpha, zorder=5))
        label(ax, s[0], s[1] - 0.28, "start", MUTED, size, alpha, ha="center")
        goal_alpha = alpha if goal_alpha is None else goal_alpha
        if goal_alpha > 0.01:
            ax.plot(*g, marker="*", color=YELLOW, markersize=17, alpha=goal_alpha, zorder=5, markeredgecolor=BG)
            label(ax, g[0], g[1] + 0.3, "goal", MUTED, size, goal_alpha, ha="center")


class Clip:
    """Frames of an existing recording, shown inside a scene."""

    def __init__(self, path, start, end, speed=1.0):
        self.capture = cv2.VideoCapture(str(path))
        self.start, self.end, self.speed = start, end, speed
        self.duration = (end - start) / speed

    def frame(self, s):
        self.capture.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * min(self.start + s * self.speed, self.end))
        ok, image = self.capture.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if ok else None

    def scene(self, title, sub=None, caption=None):
        def draw(fig, s):
            ax = canvas(fig)
            fade = ramp(s, 0, 0.6)
            heading(ax, title, sub, fade)
            image = self.frame(s)
            if image is not None:
                h, w = image.shape[:2]
                max_w, max_h = 14.8, 6.3
                scale = min(max_w / w, max_h / h)
                width, height = w * scale, h * scale
                x0, y0 = 8 - width / 2, 0.75 + (max_h - height) / 2
                ax.imshow(image, extent=[x0, x0 + width, y0, y0 + height], aspect="auto", alpha=fade, zorder=1)
                ax.add_patch(Rectangle((x0, y0), width, height, facecolor="none", edgecolor=DIM, lw=1.5, zorder=2))
                ax.set_xlim(0, 16)
                ax.set_ylim(0, 9)
            if caption:
                label(ax, 8, 0.4, caption, MUTED, 12.5, fade, ha="center")
        return draw
