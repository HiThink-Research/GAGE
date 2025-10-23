import contextlib
import io
import itertools
import multiprocessing
import os.path as osp
import signal
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Union
from typing import Iterable, Dict
import os
import json
def _eval_mbpp(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        details = {}
        with ProcessPoolExecutor() as executor:
            futures = []
            for i, (refer, pred) in enumerate(zip(references,
                                                  predictions)):
                # pred = self._process_answer(pred)
                
                programs = _process_test(refer, pred)
                future = executor.submit(execution, programs, i, 10)
                futures.append(future)
                details[str(i)] = {}
                details[str(i)]['origin'] = predictions[i]
                details[str(i)]['programs'] = programs
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures)):
                index, ret = future.result()
                result[ret] += 1
                details[str(index)]['result'] = ret
                details[str(index)]['is_correct'] = (ret == 'pass')
        result['score'] = result['pass'] / len(predictions) * 100
        result['details'] = details
        return result

def _process_test(test_cases, pred):
    formatted = pred + '\n'
    for test_case in test_cases:
        formatted += test_case 
        formatted += '\n'
    return formatted

class TimeOutException(Exception):
    pass

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return task_id, 'timeout'
    return task_id, key[0]


# def stream_jsonl(filename: str) -> Iterable[Dict]:
#     """
#     Parses each jsonl line and yields it as a dictionary
#     """
#     if filename.endswith(".gz"):
#         with open(filename, "rb") as gzfp:
#             with gzip.open(gzfp, 'rt') as fp:
#                 for line in fp:
#                     if any(not x.isspace() for x in line):
#                         yield json.loads(line)
#     else:
#         with open(filename, "r") as fp:
#             for line in fp:
#                 if any(not x.isspace() for x in line):
#                     yield json.loads(line)
                    
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def _eval_humaneval(prompts, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        details = {}
        with ProcessPoolExecutor() as executor:
            futures = []
            for i, (prom, refer, pred) in enumerate(zip(prompts, references,
                                                  predictions)):
                # pred = self._process_answer(pred)
                
                programs = prom + pred + refer+ "\n" + "check(has_close_elements)"
                print(programs)
                future = executor.submit(execution, programs, i, 10)
                futures.append(future)
                details[str(i)] = {}
                details[str(i)]['origin'] = predictions[i]
                details[str(i)]['programs'] = programs
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures)):
                index, ret = future.result()
                result[ret] += 1
                details[str(index)]['result'] = ret
                details[str(index)]['is_correct'] = (ret == 'pass')
        result['score'] = result['pass'] / len(predictions) * 100
#         result['details'] = details
        return result
    
                
if __name__ == "__main__":
    predictions = ["def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ",
    "import heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums",]
    tasks = ['0','1']
    num_samples_per_task = 1
    samples = [
        dict(task_id=task_ids, completion=predictions[int(task_ids)])
        for task_ids in tasks
    ]
    write_jsonl("samples.jsonl", samples)
    