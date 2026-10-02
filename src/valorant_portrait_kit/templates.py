"""Portable masked normalized correlation against canonical fixed portraits.

Scores are similarities, not probabilities. Geometry search handles small
detector box errors without training Agent appearance from scarce screenshots.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


class TemplateMatcher:
    def __init__(self,bank_file):
        with np.load(bank_file,allow_pickle=False) as data:
            self.names=data['names'].tolist()
            self.size=int(data['size'])
            self.mask=data['mask'].astype(bool)
            self.bank=data['bank'].astype(np.float32)
            self.shifts=data['shifts'].astype(np.float32)

    @staticmethod
    def build(folder,names,output,size=34):
        arrays=[]
        for n in names:
            with Image.open(Path(folder)/(n+'.png')) as im:
                arrays.append(np.asarray(im.convert('RGBA').resize((size,size),Image.Resampling.LANCZOS),np.float32)/255)
        arrays=np.stack(arrays)
        mask=np.min(arrays[:,:,:,3],axis=0)>.9
        mask[:2]=False; mask[-3:]=False; mask[:,:2]=False; mask[:,-2:]=False
        shifts=np.array([(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1)],np.float32)
        bank=[]
        for arr in arrays:
            gray=cv2.cvtColor(arr[:,:,:3],cv2.COLOR_RGB2GRAY)
            variants=[]
            for dx,dy in shifts:
                shifted=cv2.warpAffine(gray,np.float32([[1,0,dx],[0,1,dy]]),(size,size),flags=cv2.INTER_LINEAR)
                v=shifted[mask]; v=v-v.mean()
                variants.append(v/max(float(np.linalg.norm(v)),1e-8))
            bank.append(variants)
        np.savez_compressed(output,names=np.array(names),size=size,mask=mask,bank=np.asarray(bank),shifts=shifts)

    def _values(self,crop):
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        small=np.asarray(Image.fromarray(rgb).resize((self.size,self.size),Image.Resampling.BILINEAR),np.float32)/255
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        v=gray[self.mask]; v=v-v.mean()
        return v/max(float(np.linalg.norm(v)),1e-8)

    def _score(self,crops):
        values=np.stack([self._values(c) for c in crops])
        return (values@self.bank.reshape(-1,self.bank.shape[-1]).T).reshape(len(crops),len(self.names),len(self.shifts))

    def predict(self,crop):
        scores=self._score([crop])[0].max(axis=1)
        order=np.argsort(scores)[::-1]; cid=int(order[0])
        return {'class_id':cid,'name':self.names[cid],'correlation':float(scores[cid]),'margin':float(scores[cid]-scores[order[1]])}

    def classify_box(self,image,box,refine=True,reference=None):
        h,w=image.shape[:2]
        def clip(b):
            x1,y1,x2,y2=b
            return [max(0,min(w,round(x1))),max(0,min(h,round(y1))),max(0,min(w,round(x2))),max(0,min(h,round(y2)))]
        base=clip(box)
        if base[2]<=base[0] or base[3]<=base[1]:
            return None
        boxes=[base]
        crops=[image[base[1]:base[3],base[0]:base[2]]]
        raw=self._score(crops)
        ss=raw[0].max(axis=1); order=np.argsort(ss)[::-1]
        # Accurate detections take one comparison. Search only uncertain crops.
        if refine and (ss[order[0]]<.75 or ss[order[0]]-ss[order[1]]<.16):
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            bw,bh=box[2]-box[0],box[3]-box[1]
            if reference is not None:
                # Scoreboard portraits share a column and square size. Their
                # confident neighbors constrain the search, especially when a
                # faint portrait gets a short or overly wide detector box.
                rx,side=reference
                geometries=[(rx+dx*side,cy+dy*side,side*factor) for factor in (.9,.95,1.,1.05)
                            for dy in (-.1,-.05,0.,.05,.1) for dx in (-.025,0.,.025)]
            else:
                geometries=[(cx,cy,side*factor) for side in (min(bw,bh),(bw+bh)/2,max(bw,bh)) for factor in (.9,1.,1.1)]
            for gx,gy,s in geometries:
                b=clip([gx-s/2,gy-s/2,gx+s/2,gy+s/2])
                if b not in boxes and b[2]>b[0] and b[3]>b[1]:
                    boxes.append(b); crops.append(image[b[1]:b[3],b[0]:b[2]])
            raw=self._score(crops)
        per_class=raw.max(axis=(0,2))
        order=np.argsort(per_class)[::-1]; cid=int(order[0])
        crop_i,shift_i=np.unravel_index(np.argmax(raw[:,cid,:]),raw[:,cid,:].shape)
        chosen=boxes[crop_i]
        dx,dy=self.shifts[shift_i]
        # Shift the complete tile box with the winning template alignment.
        dx*= (chosen[2]-chosen[0])/self.size; dy*=(chosen[3]-chosen[1])/self.size
        adjusted=[max(0,min(w,chosen[0]+float(dx))),max(0,min(h,chosen[1]+float(dy))),
                  max(0,min(w,chosen[2]+float(dx))),max(0,min(h,chosen[3]+float(dy)))]
        local_scores=raw[crop_i].max(axis=1)
        local_other=max(float(local_scores[i]) for i in range(len(self.names)) if i!=cid)
        return {'class_id':cid,'name':self.names[cid],'correlation':float(per_class[cid]),
                'margin':float(per_class[cid]-local_other),'global_geometry_margin':float(per_class[cid]-per_class[order[1]]),
                'xyxy':adjusted,'geometry_candidates':len(boxes)}
