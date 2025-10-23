import json
import re

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def Ans_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'

len_cate = {'OCR':40, 'artwork':400, 'celebrity':340, 'color':60, 'count':60, 'existence':60,
            'landmark':400, 'position':60, 'posters':294, 'scene':400,'code_reasoning':40, 'commonsense_reasoning':140, 'numerical_calculation':40, 'text_translation':40,"acc":2374}

def evaluation(input_path,**kwargs):
    # import pdb; pdb.set_trace()
    corrects = []
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    count = 0
    out = []

    stats = {"acc":0}
    for d in data:
        category = d['category']
        ans = Ans_Extraction(d['predict_result'])
        score = 1 if ans == d["choices"][0]["message"]["content"][0]['text'] else 0
        d['eval_result'] = {"result":"True"} if score == 1 else {"result":"False"}

        if category in stats.keys():
            stats[category] += score
        else:
            stats[category] = score
        stats['acc'] += score
        out.append(d)

    

    scores = {}
    for k in stats:
        scores[k] = stats[k] / len_cate[k]

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )
    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base
    ret.update(scores)
    
    with open(input_path,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    return {"acc": ret['acc']}



if __name__ == "__main__":
    pass