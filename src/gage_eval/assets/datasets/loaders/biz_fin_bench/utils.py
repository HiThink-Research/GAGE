def get_file_path(local_dir, split, subset):
    filename = ''
    if split == 'stock_price_predict' and subset == 'en':
        filename = 'stock_price_predict_en.jsonl'
    elif split == 'anomaly_information_tracing' and subset == 'en':
        filename = 'anomaly_information_tracing_en.jsonl'
    elif split == 'conterfactual' and subset == 'en':
        filename = 'conterfactual_en.jsonl'
    elif split == 'event_logic_reasoning' and subset == 'en':
        filename = 'event_logic_reasoning_en.jsonl'
    elif split == 'financial_data_description' and subset == 'en':
        filename = 'financial_data_description_en.jsonl'
    elif split == 'financial_multi_turn_perception' and subset == 'en':
        filename = 'financial_multi-turn_perception_en.jsonl'
    elif split == 'financial_quantitative_computation' and subset == 'en':
        filename = 'financial_quantitative_computation_en.jsonl'
    elif split == 'user_sentiment_analysis' and subset == 'en':
        filename = 'user_sentiment_analysis_en.jsonl'
    elif split == 'stock_price_predict' and subset == 'cn':
        filename = 'stock_price_predict_cn.jsonl'
    elif split == 'anomaly_information_tracing' and subset == 'cn':
        filename = 'anomaly_information_tracing_cn.jsonl'
    elif split == 'conterfactual' and subset == 'cn':
        filename = 'conterfactual_cn.jsonl'
    elif split == 'event_logic_reasoning' and subset == 'cn':
        filename = 'event_logic_reasoning_cn.jsonl'
    elif split == 'financial_data_description' and subset == 'cn':
        filename = 'financial_data_description_cn.jsonl'
    elif split == 'financial_multi_turn_perception' and subset == 'cn':
        filename = 'financial_multi-turn_perception_cn.jsonl'
    elif split == 'financial_quantitative_computation' and subset == 'cn':
        filename = 'financial_quantitative_computation_cn.jsonl'
    elif split == 'user_sentiment_analysis' and subset == 'cn':
        filename = 'user_sentiment_analysis_cn.jsonl'        

    file_path = f'{local_dir}/{subset}/{filename}'

    return file_path