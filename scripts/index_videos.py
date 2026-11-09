#!/usr/bin/env python3
"""Deterministic filename routing. Explicit overrides win; unknown clips stay visible."""
import json, pathlib, re
from urllib.parse import quote
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXTENSIONS={'.mp4','.webm','.ogv','.m4v'}
def discover(root=ROOT):
    slots=json.loads((root/'data/video_slots.json').read_text())
    overrides_path=root/'videos/placements.json'
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    output={key:[] for key in [*slots,'additional']}
    for path in sorted((root/'videos').rglob('*')):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file(): continue
        if path.is_symlink(): raise ValueError(f'Symlink videos are unsupported: {path.name}')
        if path.stat().st_size>95*1024*1024: raise ValueError(f'{path.name} exceeds 95 MiB; compress it before publishing.')
        relative=path.relative_to(root/'videos').as_posix()
        words=set(re.findall(r'[a-z0-9]+',path.stem.lower()))
        scores={key:len(words.intersection(slot['keywords'])) for key,slot in slots.items()}
        slot=overrides.get(relative)
        if slot is None:
            slot=next((key for key in slots if path.stem.lower().startswith(key+'_')),None)
        if slot is None: slot=max(scores,key=scores.get) if max(scores.values(),default=0)>0 else 'additional'
        if slot not in output: raise ValueError(f'Unknown slot {slot!r} for {relative}')
        output[slot].append({'title':re.sub(r'[_-]+',' ',path.stem).strip(), 'src':quote(path.relative_to(root).as_posix(),safe='/')})
    return output
if __name__=='__main__': print(json.dumps(discover(),indent=2))
