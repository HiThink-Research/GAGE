import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from math import isclose

@dataclass
class NumberSpan:
    raw: str
    normalized: str
    start: int
    end: int
    kind: str            # 'cn_unit' | 'percent' | 'plain'
    scaled: bool         # 百分比是否缩放（percent_mode='scale'时为True）
    as_float: Optional[float]

@dataclass
class MatchResult:
    answer: float
    answer_str: str
    matched: bool
    matched_index: Optional[int]
    matched_raw: Optional[str]
    matched_norm: Optional[str]
    diff: Optional[float]

class NumericMatcher:
    """
    用于：文本抽取数值 + 标准化 + 与标准答案集合匹配
    - percent_mode: 'keep'（45% -> 45）或 'scale'（45% -> 0.45）
    - 容差：rel_tol（相对）、abs_tol（绝对），用于浮点匹配
    """

    _UNIT_MAP = {
        '万亿': 10**12,
        '千亿': 10**11,
        '百亿': 10**10,
        '十亿': 10**9,
        '亿':   10**8,
        '千万': 10**7,
        '百万': 10**6,
        '十万': 10**5,
        '万':   10**4,
        '千':   10**3,
        '百':   10**2,
        '十':   10**1,
    }

    _CN_UNIT_PATTERN = re.compile(
        r'([-+]?\d[\d,]*(?:\.\d+)?|\d(?:\.\d+)?(?:e[-+]?\d+)?)\s*'
        r'(万亿|千亿|百亿|十亿|亿|千万|百万|十万|万|千|百|十)'
        r'(?:\s*(?:元|块|人民币|RMB|美元|USD|港元|HKD|日元|JPY|股|人|件|台))?',
        flags=re.IGNORECASE
    )
    _PERCENT_PATTERN = re.compile(
        r'([-+]?\d[\d,]*(?:\.\d+)?|\d(?:\.\d+)?(?:e[-+]?\d+)?)\s*%',
        flags=re.IGNORECASE
    )
    _PLAIN_NUM_PATTERN = re.compile(
        r'(?<!\d)([-+]?\d[\d,]*(?:\.\d+)?|\d(?:\.\d+)?(?:e[-+]?\d+)?)(?![\d%])',
        flags=re.IGNORECASE
    )

    def __init__(self, percent_mode: str = "keep", rel_tol: float = 1e-6, abs_tol: float = 1e-3):
        # isclose(100.0, 100.00001, rel_tol=1e-6)  # True

        assert percent_mode in ("keep", "scale")
        self.percent_mode = percent_mode
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    # ---------- Public APIs ----------

    def normalize(self, s: Any) -> str:
        """标准化任意输入为数值字符串（用于独立比对/展示）"""
        return self._normalize_numeric_string(s, self.percent_mode)

    def extract(self, text: str) -> List[NumberSpan]:
        """从文本中抽取所有数值（中文单位、百分比、普通数），并标准化"""
        return self._extract_numbers(text, self.percent_mode)

    def match(self, text: str, answers: List[Any]) -> Dict[str, Any]:
        """
        在文本中匹配标准答案集合。
        返回：
          - numbers: 文本抽取的数值（带位置、类型、是否缩放百分比）
          - matches: 每个答案的匹配结果（是否命中、命中的是哪个数、差值）
          - unmatched_numbers: 没被任何答案“认领”的文本数值
        """
        extracted = self.extract(text)

        # 规范化答案为 float
        norm_answers: List[Tuple[str, Optional[float]]] = []
        for a in answers:
            a_norm = self.normalize(a)
            try:
                norm_answers.append((a_norm, float(a_norm) if a_norm != "" else None))
            except ValueError:
                norm_answers.append((a_norm, None))

        used_indices = set()
        match_results: List[MatchResult] = []

        for a_str, a_val in norm_answers:
            if a_val is None:
                match_results.append(MatchResult(
                    answer=float('nan'), answer_str=a_str, matched=False,
                    matched_index=None, matched_raw=None, matched_norm=None, diff=None
                ))
                continue

            # 优先找 isclose 命中
            best_idx = None
            best_diff = None
            for i, num in enumerate(extracted):
                if num.as_float is None or i in used_indices:
                    continue
                if isclose(a_val, num.as_float, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    diff = abs(a_val - num.as_float)
                    if best_diff is None or diff < best_diff:
                        best_idx = i
                        best_diff = diff

            if best_idx is not None:
                used_indices.add(best_idx)
                num = extracted[best_idx]
                match_results.append(MatchResult(
                    answer=a_val, answer_str=a_str, matched=True,
                    matched_index=best_idx, matched_raw=num.raw,
                    matched_norm=num.normalized, diff=best_diff
                ))
            else:
                # 若无严格命中，可选择返回最近值（便于排查）；如需严格命中可删掉该分支
                nearest_idx = None
                nearest_diff = None
                for i, num in enumerate(extracted):
                    if num.as_float is None or i in used_indices:
                        continue
                    d = abs(a_val - num.as_float)
                    if nearest_diff is None or d < nearest_diff:
                        nearest_idx = i
                        nearest_diff = d
                if nearest_idx is not None:
                    num = extracted[nearest_idx]
                    match_results.append(MatchResult(
                        answer=a_val, answer_str=a_str, matched=False,
                        matched_index=nearest_idx, matched_raw=num.raw,
                        matched_norm=num.normalized, diff=nearest_diff
                    ))
                else:
                    match_results.append(MatchResult(
                        answer=a_val, answer_str=a_str, matched=False,
                        matched_index=None, matched_raw=None,
                        matched_norm=None, diff=None
                    ))

        unmatched_numbers = [
            {
                "index": i,
                "raw": n.raw,
                "normalized": n.normalized,
                "start": n.start,
                "end": n.end,
                "kind": n.kind
            }
            for i, n in enumerate(extracted) if i not in used_indices
        ]

        return {
            "numbers": [
                {
                    "index": i,
                    "raw": n.raw,
                    "normalized": n.normalized,
                    "start": n.start,
                    "end": n.end,
                    "kind": n.kind,
                    "scaled_percent": n.scaled
                }
                for i, n in enumerate(extracted)
            ],
            "matches": [
                {
                    "answer": mr.answer,
                    "answer_str": mr.answer_str,
                    "matched": mr.matched,
                    "matched_index": mr.matched_index,
                    "matched_raw": mr.matched_raw,
                    "matched_norm": mr.matched_norm,
                    "diff": mr.diff
                }
                for mr in match_results
            ],
            "unmatched_numbers": unmatched_numbers
        }

    # ---------- Internal Helpers ----------

    def _parse_chinese_number_with_units(self, s: str) -> Optional[float]:
        s = str(s).strip()
        m = self._CN_UNIT_PATTERN.search(s)
        if not m:
            return None
        num_str, unit = m.group(1), m.group(2)
        num_str = num_str.replace(',', '')
        try:
            base = float(num_str)
            mul = self._UNIT_MAP.get(unit, 1.0)
            return base * mul
        except ValueError:
            return None

    def _normalize_numeric_string(self, s: Any, percent_mode: str) -> str:
        s = str(s).strip()

        # 1) 中文单位优先
        try:
            c = self._parse_chinese_number_with_units(s)
            if c is not None:
                return self._format_num(c)
        except Exception:
            pass

        # 2) 百分号处理
        is_percent = '%' in s
        m = re.search(r'([-+]?\d[\d,]*(?:\.\d+)?|\d(?:\.\d+)?(?:e[-+]?\d+)?)', s, flags=re.IGNORECASE)
        if not m:
            return ""

        num_str = m.group(1).replace(',', '')
        try:
            val = float(num_str)
        except ValueError:
            return ""

        if is_percent and percent_mode == "scale":
            val = val / 100.0

        return self._format_num(val)

    def _format_num(self, x: float) -> str:
        try:
            if abs(x) >= 1e6 or abs(x - round(x)) < 1e-10:
                return str(int(round(x)))
            return f"{x:.6f}".rstrip('0').rstrip('.')
        except Exception:
            return ""

    def _extract_numbers(self, text: str, percent_mode: str) -> List[NumberSpan]:
        res: List[NumberSpan] = []
        t = text
        taken = [False] * (len(t) + 1)

        # 1) 中文单位
        for m in self._CN_UNIT_PATTERN.finditer(t):
            raw = m.group(0)
            norm = self._normalize_numeric_string(raw, percent_mode)
            start, end = m.span()
            self._mark_taken(taken, start, end)
            res.append(NumberSpan(
                raw=raw, normalized=norm, start=start, end=end,
                kind='cn_unit', scaled=False,
                as_float=float(norm) if norm else None
            ))

        # 2) 百分数
        for m in self._PERCENT_PATTERN.finditer(t):
            start, end = m.span()
            if self._overlaps_taken(taken, start, end):
                continue
            raw = m.group(0)
            norm_keep  = self._normalize_numeric_string(raw, 'keep')
            norm_scale = self._normalize_numeric_string(raw, 'scale')
            norm = norm_scale if percent_mode == 'scale' else norm_keep
            self._mark_taken(taken, start, end)
            res.append(NumberSpan(
                raw=raw, normalized=norm, start=start, end=end,
                kind='percent', scaled=(percent_mode == 'scale'),
                as_float=float(norm) if norm else None
            ))

        # 3) 普通数字（排除日期/时间/分数）
        for m in self._PLAIN_NUM_PATTERN.finditer(t):
            start, end = m.span()
            if self._overlaps_taken(taken, start, end):
                continue
            raw = m.group(0)
            if self._looks_like_datetime_or_ratio(t, start, end):
                continue
            norm = self._normalize_numeric_string(raw, percent_mode)
            self._mark_taken(taken, start, end)
            res.append(NumberSpan(
                raw=raw, normalized=norm, start=start, end=end,
                kind='plain', scaled=False,
                as_float=float(norm) if norm else None
            ))

        res.sort(key=lambda x: x.start)
        return res

    @staticmethod
    def _mark_taken(taken: List[bool], s: int, e: int) -> None:
        for i in range(s, e):
            if 0 <= i < len(taken):
                taken[i] = True

    @staticmethod
    def _overlaps_taken(taken: List[bool], s: int, e: int) -> bool:
        return any(taken[i] for i in range(max(0, s), min(e, len(taken))))

    @staticmethod
    def _looks_like_datetime_or_ratio(text: str, s: int, e: int) -> bool:
        window = text[max(0, s-2):min(len(text), e+2)]
        if re.search(r'\d{1,4}\s*[-/]\s*\d{1,2}(\s*[-/]\s*\d{1,2})?', window):
            return True
        if re.search(r'\d{1,2}\s*:\s*\d{2}', window):
            return True
        if re.search(r'[年月日]', window):
            return True
        if re.search(r'\d\s*/\s*\d', window):
            return True
        return False

if __name__ == "__main__":
    matcher = NumericMatcher(percent_mode="keep", rel_tol=1e-6, abs_tol=1e-1)

    text = """
    "### 🎯 比亚迪止盈位分析  
基于同花顺实时数据和工具，比亚迪（002594.SZ）的止盈位可从技术面和基本面综合分析。核心结论：**当前止盈位参考区间为116.59-118.58元**，需结合市场动态灵活调整。以下是详细拆解：

---

### 🔍 一、技术面：关键位与均线系统  
1. **压力位（止盈参考上沿）**：  
   - 当前压力位为 **116.59元**（基于2025-09-17数据）[^8]。  
   - 若股价触及此位且封单金额充足，可能触发止盈信号。  

2. **均线支撑与阻力**：  
   - **短期均线（MA5/MA10）**：108元（2025-09-17）[^9]，股价需站稳此线方可突破压力位。  
   - **长期均线（MA120/MA250）**：113.45元（2025-09-17）[^9]，若突破此线，中期趋势转强。  

<visual>{"chart":"","query":"比亚迪短期均线、长期均线"}</visual>  

---

### ⚖️ 二、基本面：催化因素与风险  
1. **利好催化**：  
   - **新品上市**：方程豹钛7（17.98万起）[^2]、唐DM-i智驾版（17.98万起）[^5]，扩大市场渗透。  
   - **全球化布局**：1-8月海外销量超63万辆[^4]，腾势/仰望品牌加速出海。  
   - **技术前瞻**：积极布局具身智能机器人领域[^3]，潜在增长点明确。  

2. **风险提示**：  
   - **涨停强度弱**：近10日无涨停记录（0次）[^6]，显示短期动能不足。  
   - **封单数据待验证**：9月17日未涨停（涨停价118.58元）[^7]，需观察后续封单金额变化。  

<visual>{"chart":"","query":"比亚迪连续10天涨停次数"}</visual>  

---

### 💡 三、操作建议  
- **止盈策略**：  
  - **激进型**：目标压力位 **116.59元**（当前支撑位103.71元[^8]），突破后可部分止盈。  
  - **稳健型**：待股价站稳 **MA120（113.45元）** 后，跟进压力位。  
- **监控指标**：  
  - 封单金额、涨停时间（若涨停）[^7]；  
  - 海外销量数据及新品市场反馈[^4][^2]。  

---

**以上分析仅供参考，不作为投资依据。** 市场瞬息万变，建议结合实时行情与个人风险偏好决策！ 🚀"
    """
    answers = [114.06, '比亚迪', '002594.SZ']  # 对应 12.3亿、45%（按scale时是0.45）、7500万元

    # 1) 若你希望把百分数按比例缩放
    # report = match_answers_in_text(text, answers, percent_mode="scale", rel_tol=1e-6, abs_tol=1e-6)
    # print(report)
    # 2) 若你想保留百分数的原始数值（45）
    report = matcher.match(text, answers)
    import json
    with open("out.jsonl",'w') as f:
        f.write(json.dumps(report, ensure_ascii=False) + '\n')