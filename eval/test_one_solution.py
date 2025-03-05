"""
Run solutions from one problem.
"""

import re
import ast
import inspect
import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import time
import testing_util as test_util
from pathlib import Path

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict


EXAMPLE_RESULTS = {
    "0": [[-2]],
    "1": [[False, False, False]],
    "2": [[True, True]],
    "3": [
        [
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
        ]
    ],
    "4": [
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    ],
}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10


def print_results(results: Dict, args: argparse.Namespace = None):
    """
    Given the results evaluated against the testcases we output some statistics.

    >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        problem_results = np.asarray(results[index])
        res.extend(problem_results)
        per_prob_res.append(np.mean(problem_results > 0))
        all_correct.append(np.all(problem_results > 0))

    # We count both compile errors and runtime errors for multiple tests as one error.
    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])
    total_testcases = len(res)
    if args and args.debug:
        print(
            f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}"
        )
        print(
            f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}"
        )
        print(f"number of test cases run = {total_testcases}")

    test_case_average = np.mean(np.array(per_prob_res))
    strict_accuracy = np.mean(np.array(all_correct))
    print(f"Test Case Average (average accuracy over problems) = {test_case_average}")
    print(
        f"Strict Accuracy (all test cases passed / total problems) = {strict_accuracy}"
    )

    return test_case_average, strict_accuracy


# Dummy `test_util.run_test` function for debugging multiprocessing.
def run_test(problem, test, debug):
    time.sleep(1)  # Simulate some work
    return [1]  # Dummy test result


def extract_python_code(markdown_text):
    # Regex pattern to extract Python code blocks
    pattern = r"```python\n(.*?)\n```"

    # Find all matches in the markdown text
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    return matches


def get_function_body(func):
    # Get the source code of the function
    # print(f"Before extraction: {func}")
    func = extract_python_code(func)[0]
    # print(f"After extraction: {func}")
    source = inspect.getsource(func)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Extract the function definition node
    func_def = tree.body[0]

    # Extract the body of the function
    body = func_def.body

    # Convert AST nodes back to source code
    body_source = ast.get_source_segment(source, body[0])

    # Join all lines of the body, excluding any docstrings
    body_lines = []
    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            continue  # Skip docstrings
        body_lines.append(ast.get_source_segment(source, node))

    return "\n".join(body_lines)


def check_correctness(problem, generation, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(problem, generation, debug, result):
        try:
            if debug:
                print(f"Running test for problem: {problem}")
            result.append(
                test_util.run_test(problem=problem, test=generation, debug=debug)
            )
            # Useful for debugging the multiprocessing.
            # result.append(run_test(problem=problem, test=generation, debug=debug))
            if debug:
                print(f"Test completed with result: {result}")
        except Exception as e:
            if debug:
                print(f"Error in _temp_run: {e}")

    # generation = get_function_body(generation)
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(problem, generation, debug, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            print(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"Global timeout occurred, returning default result.")
    if debug:
        print(f"Final result: {result}")
    return result[0]


def eval_and_save_problems(args):
    problems = load_dataset(
        "codeparrot/apps", split=f"{args.split}", trust_remote_code=True
    )

    codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    if os.path.exists(codes_loc):
        results_loc = os.path.join(args.save, f"all_results.json")
    else:
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
    # print(codes_loc, results_loc)

    with open(codes_loc, "r") as f:
        codes = json.load(f)

    # Only do the problems that are specified.
    if args.index:
        problems = load_dataset(
            "codeparrot/apps",
            split=f"{args.split}[{args.index}]",
            trust_remote_code=True,
        )
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = load_dataset(
            "codeparrot/apps",
            split=f"{args.split}[{start}:{end}]",
            trust_remote_code=True,
        )

    if args.stop_early:
        problems = load_dataset(
            "codeparrot/apps",
            split=f"{args.split}[{start}:{args.stop_early}]",
            trust_remote_code=True,
        )

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        try:
            if isinstance(codes, dict):
                output_strings = codes[str(index + args.start)]
            else:
                output_strings = codes[index + args.start]
        except:
            # print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue

        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        sols = problem["solutions"]

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []
        if isinstance(output_strings, str):
            output_strings = [output_strings]
        for generation_idx, generation in enumerate(output_strings):
            try:
                generation = extract_python_code(generation)[0]
            except:
                generation = ""
            if args.debug:
                print(f"\nTesting solution {generation_idx}, {generation=}")
            curr_res = [-2]
            try:
                curr_res = check_correctness(
                    problem, generation=generation, timeout=TIMEOUT, debug=args.debug
                )
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        if args.debug:
            print(
                f"\nHow to read results [-2] = compile error, [-1] = runtime error, [False] = failed test case, [True] = passed test case"
            )
            # print(f"results = {res}")

        results[index + args.start + args.index] = res

        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb

                pdb.set_trace()
                print("didn't save problem due to {e}")
        # break

    return results


def load_jsonl(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            result.append(json.loads(line))
    return result


def convert_jsonl_to_json(data_jsonl):
    data_json = {}
    for dct in data_jsonl:
        data_json[dct["task_id"]] = dct["parsed_predict"]
    return data_json


def save_json(data, path):
    start, end = 0, len(data)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{start}-{end}_codes.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def save_jsonl(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    data_jsonl = load_jsonl(args.jsonl_path)
    args.start = 0
    args.end = len(data_jsonl)
    data_json = convert_jsonl_to_json(data_jsonl)
    save_json(data_json, "./results")

    if args.print_results:
        results = {}
        results_loc = os.path.join(args.save, f"all_results.json")
        if os.path.exists(results_loc):
            results_loc = os.path.join(args.save, f"all_results.json")
            print(f"------ in all results ------")
        elif os.path.exists(
            os.path.join(args.save, f"{args.start}-{args.end}_results.json")
        ):
            results_loc = os.path.join(
                args.save, f"{args.start}-{args.end}_results.json"
            )
            print(f"------ in args.start-args.end_result ------")
        else:
            print("No results to print exiting.")
            exit()

        with open(results_loc, "r") as f:
            results = json.load(f)
        print_results(results, args)
        exit()

    if not args.skip_evals:
        print(f"in eval: {args}")
        results = eval_and_save_problems(args)

    test_case_average, strict_accuracy = print_results(results, args)

    # underlay
    all_metrics = load_jsonl(f"{args.dump}/all_metrics.jsonl")
    tests = load_json("results/all_results.json")

    for idx, dct in enumerate(data_jsonl):
        cur_results = tests[str(idx)][0]
        if sum(cur_results) == len(cur_results):
            dct["matched"] = True

    for dct in all_metrics:
        if dct["dataset"] == Path(args.jsonl_path).stem:
            try:
                _ = dct["metric"].pop("accuracy")
            except:
                pass
            dct["metric"]["test_case_average"] = round(test_case_average, 4)
            dct["metric"]["accuracy"] = round(strict_accuracy, 4)

    save_jsonl(all_metrics, f"{args.dump}/all_metrics.jsonl")
    save_jsonl(data_jsonl, args.jsonl_path)


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    parser = argparse.ArgumentParser(
        description="Testing a Language Model on Python Code"
    )
    parser.add_argument("--jsonl_path", type=str, help="results in jsonl format")
    parser.add_argument("--dump", type=str, help="dump dir path (obm)")
    parser.add_argument(
        "-t",
        "--test_loc",
        default="../data_split/test.json",
        type=str,
        help="path to the json containing problem paths to be evaluated.",
    )
    parser.add_argument(
        "-r", "--root", default="../", type=str, help="where the data is stored."
    )
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument(
        "-e",
        "--end",
        default=None,
        type=int,
        help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.",
    )
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument(
        "-p",
        "--print_results",
        action="store_true",
        help="If you have already evaluated the results and only want to print them.",
    )
    parser.add_argument(
        "--skip_evals",
        action="store_true",
        help="If you want to skip the evals similar to print results.",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument(
        "--save",
        type=str,
        default="./results",
        help="Where the evaluated data is loaded from and results saved to.",
    )
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--stop-early", default=None, type=int)

    args = parser.parse_args()

    main(args)
