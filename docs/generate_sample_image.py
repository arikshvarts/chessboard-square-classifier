from PIL import Image, ImageDraw, ImageFont
import os

out_path = os.path.join('docs', 'assets', 'sample_debug_grid.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

size = 480
sq = size // 8
img = Image.new('RGB', (size, size), 'white')
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype('arial.ttf', size=12)
except Exception:
    font = ImageFont.load_default()

for r in range(8):
    for c in range(8):
        x0, y0 = c * sq, r * sq
        x1, y1 = x0 + sq, y0 + sq
        color = (181, 136, 99) if (r + c) % 2 else (240, 217, 181)
        draw.rectangle([x0, y0, x1, y1], fill=color)
        sq_name = f"{chr(ord('a') + c)}{8 - r}"
        draw.text((x0 + 4, y0 + 4), sq_name, fill=(0, 0, 0), font=font)

img.save(out_path)
print(f"Wrote {out_path}")
