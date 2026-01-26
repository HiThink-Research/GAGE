def get_file_path(local_dir, split, subset):
    filename = ''
    if split == 'stock_price_predict' and subset == 'en':
        filename = 'stock_price_predict_en.jsonl'

    file_path = f'{local_dir}/{subset}/{filename}'

    return file_path