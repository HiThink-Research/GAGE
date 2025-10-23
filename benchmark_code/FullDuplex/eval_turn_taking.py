import json
import numpy as np
import tqdm
import random

random.seed(42)


def predict_turn_taking(logits, labels, thresholds):
    """
    根据阈值预测判停的位置
    
    判停由下列4个阈值决定
    连续t_speak个token的logits小于l_speak，则判为说话片段
    连续t_pause个token的logits大于l_pause，则判为静音片段
    静音片段之前必须有说话片段
    """
    t_speak, l_speak, t_pause, l_pause = thresholds

    n_pred = 0
    n_corr = 0
    # 一个音频可能包含多个speak和pause片段
    speak_idx = -1
    pause_idx = -1
    pause_idx_gt = -1  # 当前静音片段真实结束位置（同一个静音片段内判停多次，n_corr只算一次）
    chunk_size = round(len(labels) / len(logits))

    speak_ids = (logits < l_speak).nonzero()[0][::-1].tolist()
    pause_ids = (logits > l_pause).nonzero()[0][::-1].tolist()

    def find_consecutive_ids(ids, i_min, n_consecutive):
        n = 0  # 连续序号计数
        i0 = -1
        while ids:
            i = ids.pop()
            if i < i_min:
                continue
            n += 1
            if i - i0 > 1:
                n = 1
            if n == n_consecutive:
                return i - n + 1
            i0 = i

    while True:
        speak_idx = find_consecutive_ids(speak_ids, pause_idx + 1, t_speak)  # 先搜索speak片段
        if speak_idx is not None:
            pause_idx = find_consecutive_ids(pause_ids, speak_idx + 1, t_pause)  # 在speak片段之后搜索pause
            if pause_idx is not None:
                n_pred += 1
                if (
                    pause_idx * chunk_size > pause_idx_gt
                    and labels[pause_idx * chunk_size: (pause_idx + t_pause) * chunk_size].all()
                ):
                    n_corr += 1
                    j = labels[pause_idx * chunk_size:].argmin()  # 下一个False（非静音）的位置
                    pause_idx_gt = pause_idx * chunk_size + j if j > 0 else len(labels)
            else:  # 没找到pause片段
                break
        else:  # 没找到speak片段
            break

    return n_pred, n_corr


def search_threshold_for_f1(data, logits, labels):
    """网格搜索判停阈值，最大化F1"""

    results = {}  # precision (1%) -> scores
    result_none_cache = set()
    logits_min_max = [(p.min(), p.max()) for p in logits]

    # 阈值搜索范围
    t_speak_range = (1, 11, 1)  # min, max, step
    l_speak_range = (5, -5.5, -0.5)
    t_pause_range = (1, 11, 1)
    l_pause_range = (-5, 5.5, 0.5)
    search_space = [
        (t_speak, l_speak, t_pause, l_pause)
        for t_speak in np.arange(*t_speak_range)
        for l_speak in np.arange(*l_speak_range)
        for t_pause in np.arange(*t_pause_range)
        for l_pause in np.arange(max(l_speak, l_pause_range[0]), *l_pause_range[1:])
    ]
    n_true = calc_n_true(data)

    for thresholds in tqdm.tqdm(search_space, 'searching thresholds'):
        n_pred = 0
        n_corr = 0
        for i, (d, p, t) in enumerate(zip(data, logits, labels)):
            if thresholds[1] < logits_min_max[i][0] or thresholds[3] > logits_min_max[i][1]:
                continue
            if (k := (i,) + thresholds) in result_none_cache:
                result_none_cache.remove(k)
                continue
            n_pred_i, n_corr_i = predict_turn_taking(p, t, thresholds)
            n_pred += n_pred_i
            n_corr += n_corr_i
            if n_pred_i == 0:  # 当前阈值无结果，所有更严格的阈值可跳过计算
                k = (i,) + thresholds
                for t_speak in np.arange(thresholds[0], *t_speak_range[1:]):
                    for l_speak in np.arange(thresholds[1], *l_speak_range[1:]):
                        for t_pause in np.arange(thresholds[2], *t_pause_range[1:]):
                            for l_pause in np.arange(thresholds[3], *l_pause_range[1:]):
                                result_none_cache.add((i,) + (t_speak, l_speak, t_pause, l_pause))
        try:
            prec = n_corr / n_pred
            reca = n_corr / n_true
            f1 = 2 / (1 / prec + 1 / reca)
        except ZeroDivisionError:
            prec, reca, f1 = 0., 0., 0.
        results.setdefault(int(prec * 100), []).append((thresholds, prec, reca, f1))

    r_best = None
    for k in sorted(results):
        r = max(results[k], key=lambda _r: _r[-1])
        print('thresholds={}\tprecision={:.4f} recall={:.4f} f1={:.4f}'.format(*r))
        if r_best is None or r[-1] > r_best[-1]:
            r_best = r

    return r_best


def calc_n_true(data: list[dict]) -> int:
    """返回实际需要判停的总数（ground truth）"""
    return sum(sum(
        1 for s, e in d['messages'][-1]['content'][-1]['voice_time']
        if d['messages'][-1]['content'][-1]['audio_length'] - e > 0.1  # 如果说话结束到音频结束小于0.1秒，则不需要判停
    ) for d in data)


def calc_f1_score(data, logits, labels, thresholds):
    n_true = calc_n_true(data)
    n_pred = 0
    n_corr = 0
    n_corr_list = []
    for i, (d, p, t) in enumerate(zip(data, logits, labels)):
        n_pred_i, n_corr_i = predict_turn_taking(p, t, thresholds)
        n_pred += n_pred_i
        n_corr += n_corr_i
        n_corr_list.append(n_corr_i)
    try:
        prec = n_corr / n_pred
        reca = n_corr / n_true
        f1 = 2 / (1 / prec + 1 / reca)
    except ZeroDivisionError:
        prec, reca, f1 = 0., 0., 0.
    return n_corr_list, prec, reca, f1


def evaluation(input_path, **kwargs):
    data = []
    logits = []
    labels = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for l in f:
            d = json.loads(l)
            p = np.array(d['predict_result'])
            t = np.zeros((int(d['messages'][-1]['content'][0]['audio_length'] * 1000),), dtype=np.bool_)  # resolution: 1ms
            for s, e in d['messages'][-1]['content'][0]['voice_time']:
                t[int(s * 1000):] = False
                t[int(e * 1000):] = True
            data.append(d)
            logits.append(p)
            labels.append(t)

    # 搜索判停阈值，计算F1值
    n_sample = 200  # 采样200条用于搜索阈值
    sampled_ids = random.sample(range(len(data)), n_sample) if len(data) > n_sample else list(range(len(data)))
    thresholds = search_threshold_for_f1(
        [data[i] for i in sampled_ids],
        [logits[i] for i in sampled_ids],
        [labels[i] for i in sampled_ids],
    )[0]
    n_corr_list, prec, reca, f1 = calc_f1_score(data, logits, labels, thresholds)
    print(f'{input_path}: thresholds={thresholds}\tprecision={prec:.4f} recall={reca:.4f} f1={f1:.4f}')

    with open(input_path, 'w', encoding='utf-8') as f:
        for i, d in enumerate(data):
            d['eval_result'] = {"result": n_corr_list[i]}
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    return {'recall': reca, 'precison': prec, 'f1_score': f1}


if __name__ == '__main__':
    pass
