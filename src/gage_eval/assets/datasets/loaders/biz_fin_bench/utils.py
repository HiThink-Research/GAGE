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
    
    file_path = f'{local_dir}/{subset}/{filename}'

    return file_path