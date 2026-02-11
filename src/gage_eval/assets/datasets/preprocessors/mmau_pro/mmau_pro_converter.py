"""MMAU-Pro 预处理器。

参考 HF 数据集 `gamma-lab-umd/MMAU-Pro` 与 reference 目录下的 MMSU 预处理逻辑，
将本地 jsonl 记录转换为统一的 `Sample` 模式，并在预处理阶段把音频转为 base64。

注意：
- 原始字段中 `audio_path` 为相对路径列表（例如：["data/xxxx.wav"]）
- 这里通过 `audio_path_root` 将其转为本地绝对路径后，再逐条编码为 base64 data URL
- 每次 `to_sample` 仅处理当前样本的一条音频，避免一次性批量 base64 转码过多音频
"""

from __future__ import annotations

import base64  # 用于将二进制音频转为 base64 字符串
import os  # 处理路径拼接与规范化
from typing import Any, Dict, List, Optional  # 类型注解

from loguru import logger  # 日志记录

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor  # 预处理基类
from gage_eval.assets.datasets.sample import (  # Sample 统一数据结构
    SCHEMA_VERSION,
    Message,
    MessageContent,
    Sample,
)


def _normalize_audio_paths(value: Any) -> List[str]:
    """将 MMAU-Pro 中的 `audio_path` 规范化为字符串列表。

    参数:
        value: 原始 `audio_path` 字段，可为 str / list / tuple / None 等

    返回:
        规范化后的非空字符串列表
    """
    # 如果为 None，直接返回空列表
    if value is None:
        return []
    # 如果本身是 list / tuple，则过滤掉 None 与空串
    if isinstance(value, (list, tuple)):
        return [str(p) for p in value if p not in (None, "")]
    # 其他情况统一转为单元素列表
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _build_audio_file_path(root: Optional[str], rel_path: Optional[str]) -> str:
    """根据根目录与相对路径构建本地音频绝对路径。

    参数:
        root: 音频根目录，例如 `/mnt/.../GAGE/mmau_pro_data`
        rel_path: 记录中的相对路径，例如 `data/xxxx.wav`

    返回:
        本地音频的绝对路径

    异常:
        ValueError: 当 root 或 rel_path 缺失时抛出
    """
    # 根目录或相对路径为空视为配置错误，抛出异常
    if root is None or rel_path is None:
        raise ValueError("audio_path_root 或 audio 相对路径为空，请检查配置与数据。")
    # 去掉相对路径开头的 `/`，避免 `os.path.join` 覆盖根目录
    rel_norm = rel_path.lstrip("/").strip()
    # 拼接为完整路径
    full = os.path.join(root, rel_norm)
    # 规范化为绝对路径，便于排查问题
    return os.path.abspath(full)


def _encode_audio_to_base64(path: str) -> str:
    """读取本地音频文件并转为 base64 字符串。

    参数:
        path: 本地音频绝对路径

    返回:
        不带 data URL 前缀的 base64 字符串
    """
    # 以二进制方式读取整个文件内容
    with open(path, "rb") as f:
        data = f.read()
    # 将二进制内容编码为 base64，并转为 UTF-8 字符串
    return base64.b64encode(data).decode("utf-8")


class MMAUProConverter(BasePreprocessor):
    """将 MMAU-Pro 记录转换为 GAGE `Sample` 格式。"""

    def to_sample(
        self,
        record: Dict[str, Any],
        schema_version: str = SCHEMA_VERSION,
        audio_path_root: Optional[str] = None,
        audio_index: int = 0,
        **kwargs: Any,
    ) -> Sample:
        """将一条原始记录转换为 `Sample`。

        参数:
            record: 单条 MMAU-Pro jsonl 记录
            schema_version: Sample 使用的 schema 版本
            audio_path_root: 本地音频根目录，必填
            audio_index: 当存在多段音频时选用的下标，默认第一段

        返回:
            填充完成的 `Sample` 实例
        """
        try:
            # 保留一份可变副本
            sample: Dict[str, Any] = dict(record)

            # 基础字段
            sample_id = str(sample.get("id") or "")
            question = str(sample.get("question") or "").strip()
            answer = sample.get("answer")
            answer_text = "" if answer is None else str(answer)

            # 选项列表（MCQ 时非空）
            raw_choices = sample.get("choices") or []
            if isinstance(raw_choices, (list, tuple)):
                choices = [str(c) for c in raw_choices if c is not None]
            else:
                choices = [str(raw_choices)] if raw_choices not in ("", None) else []

            # 解析音频路径列表，并根据 audio_index 选定一条
            audio_paths = _normalize_audio_paths(sample.get("audio_path"))
            picked_audio: Optional[str] = None
            if audio_paths and 0 <= audio_index < len(audio_paths):
                picked_audio = audio_paths[audio_index]

            # 构建 base64 音频片段：一次只对当前选择的一条音频做 base64 转码
            audio_frag: Optional[MessageContent] = None
            if picked_audio:
                audio_file_path = _build_audio_file_path(audio_path_root, picked_audio)
                if not os.path.isfile(audio_file_path):
                    raise FileNotFoundError(f"未找到音频文件: {audio_file_path}")
                # 只针对当前样本的一条音频做 base64 转码，避免批量处理过多音频
                b64 = _encode_audio_to_base64(audio_file_path)
                # 统一使用 wav 媒体类型；如果存在 mp3 等，可后续按需要扩展
                audio_url = f"data:audio/wav;base64,{b64}"
                audio_frag = MessageContent(
                    type="audio_url",
                    audio_url={"url": audio_url},
                )

            # 构造文本提示：问题 + （可选）选项
            prompt_parts: List[str] = []
            if question:
                prompt_parts.append(question)
            if choices:
                # 将所有选项逐行展示，便于模型对齐答案
                option_lines = "\n".join(f"- {c}" for c in choices)
                prompt_parts.append(f"Choices:\n{option_lines}")
            prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

            # 组装消息内容：先音频，后文本
            user_contents: List[MessageContent] = []
            if audio_frag is not None:
                user_contents.append(audio_frag)
            # 文本提示始终加入，哪怕为空也维持统一结构
            user_contents.append(MessageContent(type="text", text=prompt))

            messages = [
                Message(
                    role="user",
                    content=user_contents,
                )
            ]

            # 元信息：携带 MMAU-Pro 里常见的辅助字段，便于后续过滤与分析
            metadata: Dict[str, Any] = {
                # 原始语义标签与任务信息
                "category": sample.get("category"),
                "length_type": sample.get("length_type"),
                "perceptual_skills": sample.get("perceptual_skills"),
                "reasoning_skills": sample.get("reasoning_skills"),
                "task_classification": sample.get("task_classification"),
                "task_identifier": sample.get("task_identifier"),
                "kwargs": sample.get("kwargs"),
                # 文本内容与参考答案，方便后续 metric（特别是官方综合评估封装）
                "question": question,
                "answer": answer_text,
                "transcription": sample.get("transcription"),
                "sub_cat": sample.get("sub-cat"),
                # 音频路径相关信息
                "audio_paths": audio_paths,
                "picked_audio": picked_audio,
                "audio_path_root": audio_path_root,
            }

            # 参考答案与标签：统一使用文本形式
            references = [answer_text] if answer_text else []
            label = answer_text

            # 如果 id 缺失，用 question 的哈希兜底，保持稳定性
            if not sample_id:
                sample_id = str(hash(question or str(sample)))

            return Sample(
                id=sample_id,
                schema_version=schema_version,
                messages=messages,
                options=choices or None,
                references=references,
                label=label,
                metadata=metadata,
            )
        except Exception as exc:
            # 出错时记录详细日志并抛出，方便上层感知失败样本
            logger.error("MMAU-Pro 转换 Sample 失败: {}", exc)
            raise


__all__ = ["MMAUProConverter"]

