"""Assemble the README GIF from the project's existing recorded MAZE frames.

Requires Pillow. Run from any directory; no model inference or new drawing.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
site = root / 'docs'
source = (site / 'static/js/cases.js').read_text(encoding='utf-8')
cases = json.loads(source[source.index('{'):source.rfind('}')+1])
case = next(c for c in cases['verse'] if c['id'] == 'maze_4568')

def font(size, bold=False):
    for filename in (('arialbd.ttf' if bold else 'arial.ttf'),
                     ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')):
        for path in (Path('C:/Windows/Fonts') / filename, Path(filename)):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)

def frame(image_path, label, footer):
    canvas = Image.new('RGB', (640, 760), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 18), 'VERSE | Recorded MAZE rollout', font=font(25, True), fill='#29233b')
    draw.text((26, 60), label, font=font(22), fill='#5b3fa8')
    # Preserve the published tool-output pixels at their original resolution.
    image = Image.open(site / image_path).convert('RGB')
    if image.size != (588, 588):
        raise ValueError(f'Unexpected frame size: {image_path}')
    canvas.paste(image, (26, 100))
    draw.text((26, 714), footer, font=font(20), fill='#29233b')
    return canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT)

frames = [frame(case['inputs'][0]['src'], 'Input: follow the four actions',
                'Actions: left > down > down > left')]
for index, step in enumerate(case['steps'], 1):
    action = json.loads(step['json'])[0]['label']
    frames.append(frame(step['images'][0], f'Step {index}/4 | Path Tracer: {action}',
                        'Actions: left > down > down > left'))
frames.append(frame(case['steps'][-1]['images'][0], 'Final answer',
                    f"Prediction: {case['final']['answer']}    |    Ground truth: {case['gt']}"))
output = site / 'static/images/readme_maze_demo.gif'
frames[0].save(output, save_all=True, append_images=frames[1:],
               duration=[1400, 1100, 1100, 1100, 1100, 2600], loop=0, disposal=2)
print(f'Created {output.name}: {len(frames)} frames, {output.stat().st_size} bytes')
