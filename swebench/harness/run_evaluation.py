from __future__ import annotations

import docker
import json
import platform
import traceback
import math
import os
import time

if platform.system() == 'Linux':
    import resource

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from tqdm import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )


# D* Suspiciousness Computation Functions
def compute_dstar_suspiciousness(coverage_matrix, test_results, star_power=2):
    """
    Compute D* suspiciousness scores for program statements

    Args:
        coverage_matrix: 2D array where coverage_matrix[i][j] indicates if test i covered statement j
        test_results: Array where test_results[i] is 1 if test i failed, 0 if passed
        star_power: The * value in D* (default=2)

    Returns:
        Array of suspiciousness scores for each statement
    """
    if not coverage_matrix:
        return []
    num_statements = len(coverage_matrix[0])
    suspiciousness = []

    for stmt in range(num_statements):
        ncf = nuf = ncs = 0  # Initialize counters
        for test in range(len(test_results)):
            if coverage_matrix[test][stmt] == 1:  # Statement covered by test
                if test_results[test] == 1:  # Test failed
                    ncf += 1
                else:
                    ncs += 1
            elif test_results[test] == 1:  # Not covered, test failed
                nuf += 1

        # Compute D* score
        if (nuf + ncs) == 0:
            score = 0
        else:
            score = math.pow(ncf, star_power) / (nuf + ncs)

        suspiciousness.append(score)

    return suspiciousness


def rank_statements(suspiciousness):
    """
    Rank statements by suspiciousness score from highest to lowest

    Args:
        suspiciousness: Array of suspiciousness scores

    Returns:
        Array of statement indices sorted by descending suspiciousness
    """
    ranked = sorted(enumerate(suspiciousness), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked]


def extract_test_results_from_output(test_output, logger):
    """
    Parses test output to extract test results.

    Args:
        test_output: The output from running the tests.
        logger: Logger for logging messages.

    Returns:
        Dictionary mapping test names to pass/fail (0 for pass, 1 for fail)
    """
    test_results = {}
    try:
        lines = test_output.split("\n")
        collecting_tests = False
        for line in lines:
            line = line.strip()
            if line.startswith("collected"):
                collecting_tests = True
                continue
            if collecting_tests:
                if line.startswith("test_"):
                    parts = line.split()
                    test_name = parts[0]
                    # Determine if passed or failed
                    result = 1
                    for p in parts:
                        if p == "PASSED":
                            result = 0
                            break
                        elif p == "FAILED":
                            result = 1
                            break
                    test_results[test_name] = result
                elif line.startswith("=" * 10):
                    collecting_tests = False
        return test_results
    except Exception as e:
        logger.error(f"Error extracting test results: {str(e)}")
        return {}


def extract_coverage_matrix_and_test_results(
    coverage_data, test_results_dict, logger
):
    """
    Extracts the coverage matrix and test results from coverage data with contexts.

    Args:
        coverage_data: The parsed coverage.json data.
        test_results_dict: Dictionary of test results (0 for pass, 1 for fail)
        logger: Logger for logging messages.

    Returns:
        Tuple of (coverage_matrix, test_results)
    """
    try:
        files = coverage_data.get("files", {})
        statements = set()
        contexts = list(test_results_dict.keys())

        for file_data in files.values():
            executed = file_data.get("executed_lines", [])
            missing = file_data.get("missing_lines", [])
            statements.update(map(int, executed))
            statements.update(map(int, missing))

        statements = sorted(statements)
        stmt_to_index = {stmt: idx for idx, stmt in enumerate(statements)}
        context_to_index = {ctx: idx for idx, ctx in enumerate(contexts)}
        coverage_matrix = [[0] * len(statements) for _ in range(len(contexts))]
        test_results = [0] * len(contexts)

        for file_data in files.values():
            for context_name, context_data in file_data.get("contexts", {}).items():
                if context_name not in context_to_index:
                    continue
                ctx_idx = context_to_index[context_name]
                test_results[ctx_idx] = test_results_dict.get(context_name, 1)
                for line in context_data.get('executed_lines', []):
                    stmt_idx = stmt_to_index.get(int(line))
                    if stmt_idx is not None:
                        coverage_matrix[ctx_idx][stmt_idx] = 1

        return coverage_matrix, test_results
    except Exception as e:
        logger.error(f"Error extracting coverage data: {str(e)}")
        return None, None


def compute_dstar_from_coverage(coverage_file_path, test_results_dict, logger):
    """
    Parses coverage.json to compute D* suspiciousness scores.
    """
    try:
        with open(coverage_file_path, "r") as f:
            coverage_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading coverage data: {str(e)}")
        return None, None

    coverage_matrix, test_results = extract_coverage_matrix_and_test_results(
        coverage_data, test_results_dict, logger
    )

    if coverage_matrix is not None and test_results is not None:
        suspiciousness_scores = compute_dstar_suspiciousness(
            coverage_matrix, test_results
        )
        ranked_statements = rank_statements(suspiciousness_scores)
        return suspiciousness_scores, ranked_statements
    else:
        logger.error("Coverage matrix or test results are empty.")
        return None, None


def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):
    """
    Run a single instance with the given prediction, including D* suspiciousness computation.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            pass
    log_file = log_dir / LOG_INSTANCE

    # Set up report file + logger
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    container = None
    try:
        # Build + start instance container
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = log_dir / "patch.diff"
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container
        val = container.exec_run(
            f"git apply --allow-empty -v {DOCKER_PATCH}",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        if val.exit_code != 0:
            logger.info(f"Failed to apply patch to container, trying again...")
            val = container.exec_run(
                f"patch --batch --fuzz=5 -p1 -i {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code != 0:
                logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
                raise EvaluationError(
                    instance_id,
                    f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                    logger,
                )
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git diff", workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = log_dir / "eval.sh"
        eval_sh_content = """#!/bin/bash
# Install necessary packages
pip install --no-cache-dir coverage pytest pytest-cov

# Run tests with coverage and context tracking
coverage erase
coverage run --context=test --branch -m pytest tests/
coverage json -o coverage.json
"""
        eval_file.write_text(eval_sh_content)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))

        # Make eval.sh executable
        container.exec_run("chmod +x /eval.sh", workdir=DOCKER_WORKDIR, user=DOCKER_USER)

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Copy coverage data from container
        coverage_file_host = log_dir / "coverage.json"
        coverage_file_container = PurePosixPath(DOCKER_WORKDIR) / "coverage.json"
        bits, stat = container.get_archive(str(coverage_file_container))
        with open(coverage_file_host, "wb") as f:
            for chunk in bits:
                f.write(chunk)
        logger.info(f"Coverage data copied to {coverage_file_host}")

        # Extract test results from test output
        test_results = extract_test_results_from_output(test_output, logger)
        if not test_results:
            logger.error("No test results extracted.")

        # Compute D* suspiciousness scores
        suspiciousness_scores, ranked_statements = compute_dstar_from_coverage(
            coverage_file_host, test_results, logger
        )

        # Save D* scores as JSON
        dstar_report_path = log_dir / "dstar_report.json"
        dstar_data = {
            "suspiciousness_scores": suspiciousness_scores if suspiciousness_scores else [],
            "ranked_statements": ranked_statements if ranked_statements else []
        }
        with open(dstar_report_path, "w") as dstar_f:
            json.dump(dstar_data, dstar_f, indent=4)

        if suspiciousness_scores:
            logger.info("D* Suspiciousness Scores computed successfully.")
            logger.info("Top 10 ranked statements by suspiciousness:")
            for idx in ranked_statements[:10]:
                logger.info(f"Statement {idx}: Score = {suspiciousness_scores[idx]}")
        else:
            logger.error(f"Failed to compute D* suspiciousness scores for {instance_id}.")

        # Get git diff after running eval script
        git_diff_output_after = (
            container.exec_run("git diff", workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info(f"Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances))

    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs
            }
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")


def get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        run_id: str,
        exclude_completed: bool = True
    ):
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
    
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None}
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]
    return dataset


def make_run_report(
        predictions: dict,
        full_dataset: list,
        client: docker.DockerClient,
        run_id: str
    ) -> Path:
    completed_ids = set()
    resolved_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unresolved_ids = set()
    incomplete_ids = set()
    empty_patch_ids = set()

    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        if prediction.get(KEY_PREDICTION, None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance_id)
            report = json.loads(report_file.read_text())
            if report[instance_id]["resolved"]:
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)
        else:
            error_ids.add(instance_id)

    images = list_images(client)
    test_specs = list(map(make_test_spec, full_dataset))
    for spec in test_specs:
        image_name = spec.instance_image_key
        if image_name in images:
            unremoved_images.add(image_name)
    containers = client.containers.list(all=True)
    for c in containers:
        if run_id in c.name:
            unstopped_containers.add(c.name)

    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_dataset}
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(set(predictions.keys()) & dataset_ids)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    print(f"Unstopped containers: {len(unstopped_containers)}")
    print(f"Unremoved images: {len(unremoved_images)}")

    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }
    report_file = Path(
        list(predictions.values())[0][KEY_MODEL].replace("/", "__")
        + f".{run_id}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")
    return report_file


def get_gold_predictions(dataset_name: str, split: str):
    dataset = load_swebench_dataset(dataset_name, split)
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            KEY_PREDICTION: datum["patch"],
            KEY_MODEL: "gold",
        } for datum in dataset
    ]


def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    assert len(run_id) > 0, "Run ID must be provided"
    if platform.system() == 'Linux':
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)
    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    clean_images(client, existing_images, cache_level, clean)
    make_run_report(predictions, full_dataset, client, run_id)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout (in seconds) for running tests for each instance")
    parser.add_argument("--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images")
    parser.add_argument("--cache_level", type=str, choices=["none", "base", "env", "instance"], default="env",
                        help="Cache level - remove images above this level")
    parser.add_argument("--clean", type=str2bool, default=False, help="Clean images above cache level")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))

