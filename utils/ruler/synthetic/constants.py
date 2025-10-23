"""
Add a new task (required arguments):

TASK_NAME: {
    'tokens_to_generate': how many tokens we want to generate.
    'template': the template with at least {context} and {query}.
}
"""

TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are"""
    },
    
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """
    },
    
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:"""
    },
    
    'freq_words_extraction' : {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:"""
    },

    'qa': {
        'tokens_to_generate': 32, 
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
    },
}

# Adapted from https://github.com/hsiehjackson/RULER/tree/dbc6a83c2f60d034dfaab8e7af42dde1a5d2e3dc/scripts/eval/synthetic/constants.py
"""
Add a new task:

TASK_NAME: {
    'metric_fn': the metric function with input (predictions: [str], references: [[str]]) to compute score.
}
"""


def string_match_part(pred, ref):
    score = max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
    return score

def string_match_all(pred, ref):
    score = sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
    return score
    

TASKS_EVAL = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
