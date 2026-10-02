"""Automatic scoreboard annotation: ONNX localization + fixed PNG templates."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .detector import create_session, predict
from .templates import TemplateMatcher


class Pipeline:
    def __init__(self,bundle):
        self.bundle=Path(bundle)
        self.config=json.loads((self.bundle/'pipeline.json').read_text(encoding='utf-8'))
        self.names=json.loads((self.bundle/'classes.json').read_text())
        self.detector_names=self.config['detector_names']
        self.session=create_session(self.bundle/self.config['detector'])
        self.matcher=TemplateMatcher(self.bundle/self.config['templates'])
        assert self.names==self.matcher.names,'Template names and label indices differ'

    def predict(self,image):
        cfg=self.config
        proposals,ms=predict(self.session,image,self.detector_names,cfg['detector_confidence'])
        start=time.perf_counter()
        accepted,rejected=[],[]
        reference=None
        if cfg.get('align_scoreboard_boxes',False):
            square=[p for p in proposals if p['confidence']>=.4 and .8<=(p['xyxy'][2]-p['xyxy'][0])/max(p['xyxy'][3]-p['xyxy'][1],1e-6)<=1.25]
            if len(square)>=4:
                sx=np.array([(p['xyxy'][0]+p['xyxy'][2])/2 for p in square])
                sides=np.array([(p['xyxy'][2]-p['xyxy'][0]+p['xyxy'][3]-p['xyxy'][1])/2 for p in square])
                radius=float(np.median(sides))*.6
                groups=[np.abs(sx-c)<=radius for c in sx]
                group=max(groups,key=lambda g:int(g.sum()))
                if int(group.sum())>=4:
                    reference=(float(np.median(sx[group])),float(np.median(sides[group])))
        for p in proposals:
            t=self.matcher.classify_box(image,p['xyxy'],cfg.get('refine_boxes',True),reference)
            if t is None:
                continue
            d={**t,'confidence':p['confidence'],'detector_confidence':p['confidence'],'detector_xyxy':p['xyxy']}
            if t['correlation']>=cfg['correlation_min'] and t['margin']>=cfg['margin_min']:
                accepted.append(d)
            else:
                d['reason']='template_uncertain'; rejected.append(d)
        # The task's labels cover the scoreboard column. HUD portraits outside
        # that column must not become extra training annotations.
        if cfg.get('scoreboard_column_filter',True) and len(accepted)<4:
            for d in accepted:
                d['reason']='insufficient_scoreboard_support'; rejected.append(d)
            accepted=[]
        if cfg.get('scoreboard_column_filter',True) and len(accepted)>=4:
            centers=np.array([(d['xyxy'][0]+d['xyxy'][2])/2 for d in accepted])
            sizes=np.array([d['xyxy'][2]-d['xyxy'][0] for d in accepted])
            side=float(np.median(sizes))
            groups=[np.abs(centers-c)<=side*.6 for c in centers]
            best=max(groups,key=lambda g:(int(g.sum()),sum(accepted[i]['correlation'] for i in np.flatnonzero(g))))
            if int(best.sum())>=4:
                keep=[]
                for i,d in enumerate(accepted):
                    if best[i] and .55*side<=sizes[i]<=1.6*side:
                        keep.append(d)
                    else:
                        d['reason']='outside_scoreboard_column'; rejected.append(d)
                accepted=keep
        # Refining candidates can bring neighboring YOLO proposals onto the
        # same tile. Keep the strongest fixed-template match for each tile.
        from .metrics import iou
        unique=[]
        for d in sorted(accepted,key=lambda d:(d['correlation'],d['detector_confidence']),reverse=True):
            if any(iou(d['xyxy'],u['xyxy'])>cfg.get('refined_nms_iou',.45) for u in unique):
                d['reason']='duplicate_refined_box'; rejected.append(d)
            else:
                unique.append(d)
        unique.sort(key=lambda d:(d['xyxy'][1],d['xyxy'][0]))
        return {'detections':unique,'rejected':rejected,'proposals':proposals,'detector_ms':ms,
                'classification_ms':(time.perf_counter()-start)*1000}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[2] / 'models' / 'scoreboard')
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('annotations'))
    args=ap.parse_args()
    pipeline=Pipeline(args.bundle)
    if args.source.resolve()==args.output.resolve():
        raise ValueError('The output directory must differ from the source directory')
    files=[args.source] if args.source.is_file() else sorted(p for p in args.source.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and not p.resolve().is_relative_to(args.output.resolve()))
    assert files, 'No input images found'
    targets=set()
    for f in files:
        relative=f.relative_to(args.source) if args.source.is_dir() else Path(f.name)
        key=str(relative.with_suffix('')).casefold()
        if key in targets:
            raise ValueError(f'Images share a label stem: {relative}; give them different filenames before annotating')
        targets.add(key)
    for f in files:
        image=cv2.imdecode(np.fromfile(str(f),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {f}')
        start=time.perf_counter(); result=pipeline.predict(image)
        result.update({'source':str(f),'processing_ms':(time.perf_counter()-start)*1000})
        relative=f.relative_to(args.source) if args.source.is_dir() else Path(f.name)
        dest=args.output/relative.with_suffix('')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.with_name(dest.name+'.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        labels=[]
        for d in result['detections']:
            x1,y1,x2,y2=d['xyxy']
            labels.append(f"{d['class_id']} {(x1+x2)/(2*image.shape[1]):.8f} {(y1+y2)/(2*image.shape[0]):.8f} {(x2-x1)/image.shape[1]:.8f} {(y2-y1)/image.shape[0]:.8f}")
            a,b,c,e=[round(z) for z in d['xyxy']]
            cv2.rectangle(image,(a,b),(c,e),(50,230,80),1)
            cv2.putText(image,d['name'],(c+3,b+12),cv2.FONT_HERSHEY_SIMPLEX,.45,(50,230,80),1,cv2.LINE_AA)
        dest.with_name(dest.name+'.txt').write_text('\n'.join(labels)+('\n' if labels else ''),encoding='utf-8')
        ok,encoded=cv2.imencode('.jpg',image); assert ok
        encoded.tofile(str(dest.with_name(dest.name+'.annotated.jpg')))
        print(f'{relative}: {len(labels)} accepted, {len(result["rejected"])} rejected, {result["processing_ms"]:.1f} ms')

if __name__=='__main__':
    main()
