def get_task_type(origin_id):
    """
    获取数据所属的能力维度以及中英文版本
    """
    # 获取语言
    if "_zh.json" in origin_id:
        lang_type = "zh"
    else:
        lang_type = "en"

    # 获取 task_type, prompt_type
    if origin_id.startswith("reason_retrieve_understand_"):
        task_type = "rru"
        prompt_type = "json"
    elif origin_id.startswith("instruct_"):
        task_type = "instruct"
        prompt_type = "json"
    elif origin_id.startswith("plan_"):
        task_type = "plan"
        if "plan_json" in origin_id:
            prompt_type = "json"
        else:
            prompt_type = "str"
    elif origin_id.startswith("reason_"):
        task_type = "reason"
        prompt_type = "str"
    elif origin_id.startswith("retrieve_"):
        task_type = "retrieve"
        prompt_type = "str"
    elif origin_id.startswith("review_"):
        task_type = "review"
        prompt_type = "str"
    elif origin_id.startswith("understand_"):
        task_type = "understand"
        prompt_type = "str"

    return task_type, lang_type, prompt_type
