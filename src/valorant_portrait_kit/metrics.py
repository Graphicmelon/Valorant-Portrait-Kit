"""Class-aware counts on YOLO labels; deliberately independent of the training stack."""
def iou(a,b):
    left,top = max(a[0],b[0]),max(a[1],b[1])
    right,bottom = min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,right-left)*max(0,bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0


def match_detections(detections,boxes,w,h,names):
    truths = [{'name':names[int(c)],'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]} for c,x,y,bw,bh in boxes]
    pairs = sorted([(iou(d['xyxy'],t['xyxy']),di,ti) for di,d in enumerate(detections) for ti,t in enumerate(truths)],reverse=True)
    matched_d,matched_t,correct,mistakes = set(),set(),0,[]
    for overlap,di,ti in pairs:
        if overlap<0.5:
            break
        if di in matched_d or ti in matched_t:
            continue
        matched_d.add(di)
        matched_t.add(ti)
        if detections[di]['name']==truths[ti]['name']:
            correct+=1
        else:
            mistakes.append({'expected':truths[ti]['name'],'predicted':detections[di]['name'],'confidence':detections[di]['confidence'],'iou':overlap})
    return {'ground_truth':len(truths),'predictions':len(detections),'localized':len(matched_t),'correct_name_and_box':correct,
        'missed':[truths[i]['name'] for i in range(len(truths)) if i not in matched_t],
        'unmatched_predictions':[detections[i] for i in range(len(detections)) if i not in matched_d],'classification_mistakes':mistakes}
