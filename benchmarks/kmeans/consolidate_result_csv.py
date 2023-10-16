import hashlib
from io import BytesIO
from itertools import zip_longest
from operator import attrgetter

import numpy as np
import pandas as pd
from pandas.io.parsers.readers import STR_NA_VALUES

GOOGLE_WORKSHEET_NAME = "k-means"

DATES_FORMAT = "%Y-%m-%d"

BENCHMARK_DEFINING_COLUMNS = [
    "objective_objective_param___name",
    "objective_dataset_param___name",
    "objective_dataset_param_n_samples",
    "objective_dataset_param_n_features",
    "objective_dataset_param_dtype",
    "objective_dataset_param_random_state",
    "objective_objective_param_n_clusters",
    "objective_objective_param_init",
    "objective_objective_param_n_init",
    "objective_objective_param_max_iter",
    "objective_objective_param_tol",
    "objective_objective_param_verbose",
    "objective_objective_param_algorithm",
    "objective_objective_param_random_state",
    "objective_objective_param_sample_weight",
]

BENCHMARK_DEFINING_COLUMNS = sorted(BENCHMARK_DEFINING_COLUMNS)
_benchmark_defining_columns_identifier = "".join(sorted(BENCHMARK_DEFINING_COLUMNS))

BACKEND_PROVIDER = "Backend provider"
COMMENT = "Comment"
COMPUTE_DEVICE = "Compute device"
COMPUTE_RUNTIME = "Compute runtime"
DATA_RANDOM_STATE = "Data random state"
DATA_SAMPLE_WEIGHTS = "Data sample weights"
DTYPE = "Dtype"
INIT_TYPE = "Init type"
NB_CLUSTERS = "Nb clusters"
NB_DATA_FEATURES = "Nb data features"
NB_DATA_SAMPLES = "Nb data samples"
PLATFORM = "Platform"
PLATFORM_ARCHITECTURE = "Platform architecture"
PLATFORM_RELEASE = "Platform release"
SYSTEM_CPUS = "Nb cpus"
SYSTEM_PROCESSOR = "Cpu name"
SYSTEM_RAM = "RAM (GB)"
SYSTEM_GPU = "Gpu name"
RESULT_NB_ITERATIONS = "Result nb iterations"
RESULT_INERTIA = "Result inertia"
VERSION_INFO = "Version info"
RUN_DATE = "Run date"
SOLVER_RANDOM_STATE = "Solver random state"
WALLTIME = "Walltime"

BENCHMARK_ID_NAME = "Benchmark id"

TABLE_DISPLAY_ORDER = [
    BENCHMARK_ID_NAME,
    DTYPE,
    NB_DATA_SAMPLES,
    NB_DATA_FEATURES,
    NB_CLUSTERS,
    INIT_TYPE,
    DATA_SAMPLE_WEIGHTS,
    WALLTIME,
    BACKEND_PROVIDER,
    COMPUTE_DEVICE,
    COMPUTE_RUNTIME,
    SYSTEM_CPUS,
    SYSTEM_PROCESSOR,
    SYSTEM_GPU,
    SYSTEM_RAM,
    PLATFORM,
    PLATFORM_ARCHITECTURE,
    PLATFORM_RELEASE,
    RUN_DATE,
    VERSION_INFO,
    COMMENT,
    RESULT_NB_ITERATIONS,
    RESULT_INERTIA,
    DATA_RANDOM_STATE,
    SOLVER_RANDOM_STATE,
]

COLUMNS_DTYPES = {
    BENCHMARK_ID_NAME: str,
    DTYPE: str,
    NB_DATA_SAMPLES: np.int64,
    NB_DATA_FEATURES: np.int64,
    NB_CLUSTERS: np.int64,
    INIT_TYPE: str,
    DATA_SAMPLE_WEIGHTS: str,
    WALLTIME: np.float64,
    BACKEND_PROVIDER: str,
    COMPUTE_DEVICE: str,
    COMPUTE_RUNTIME: str,
    RESULT_NB_ITERATIONS: np.int64,
    RESULT_INERTIA: np.float64,
    PLATFORM: str,
    PLATFORM_ARCHITECTURE: str,
    PLATFORM_RELEASE: str,
    SYSTEM_CPUS: np.int64,
    SYSTEM_PROCESSOR: str,
    SYSTEM_GPU: str,
    SYSTEM_RAM: np.int64,
    DATA_RANDOM_STATE: np.int64,
    SOLVER_RANDOM_STATE: np.int64,
    VERSION_INFO: str,
    RUN_DATE: str,
    COMMENT: str,
}

COLUMNS_WITH_NONE_STRING = [DATA_SAMPLE_WEIGHTS]

# If all those fields have equal values for two given benchmarks, then the oldest
# benchmark (given by RUN_DATE) will be discarded
UNIQUE_BENCHMARK_KEY = [
    BENCHMARK_ID_NAME,
    DTYPE,
    NB_DATA_SAMPLES,
    NB_DATA_FEATURES,
    NB_CLUSTERS,
    INIT_TYPE,
    DATA_SAMPLE_WEIGHTS,
    BACKEND_PROVIDER,
    COMPUTE_DEVICE,
    COMPUTE_RUNTIME,
    PLATFORM,
    PLATFORM_ARCHITECTURE,
    SYSTEM_PROCESSOR,
    SYSTEM_CPUS,
    SYSTEM_GPU,
    DATA_RANDOM_STATE,
    SOLVER_RANDOM_STATE,
]

# Importance and say if ascending / descending
ROW_SORT_ORDER = [
    (DTYPE, True),
    (NB_DATA_SAMPLES, False),
    (NB_DATA_FEATURES, False),
    (NB_CLUSTERS, False),
    (INIT_TYPE, False),
    (DATA_SAMPLE_WEIGHTS, True),
    (WALLTIME, True),
    (BACKEND_PROVIDER, True),
    (COMPUTE_DEVICE, True),
    (COMPUTE_RUNTIME, True),
    (RESULT_NB_ITERATIONS, True),
    (RESULT_INERTIA, False),
    (SYSTEM_GPU, True),
    (SYSTEM_CPUS, True),
    (PLATFORM, True),
    (PLATFORM_ARCHITECTURE, True),
    (PLATFORM_RELEASE, True),
    (SYSTEM_PROCESSOR, True),
    (SYSTEM_RAM, True),
    (DATA_RANDOM_STATE, True),
    (SOLVER_RANDOM_STATE, True),
    (RUN_DATE, False),
    (VERSION_INFO, False),
    (COMMENT, True),
    (BENCHMARK_ID_NAME, True),
]
_row_sort_by, _row_sort_ascending = map(list, zip(*ROW_SORT_ORDER))

PARQUET_TABLE_DISPLAY_MAPPING = dict(
    time=WALLTIME,
    objective_value=RESULT_INERTIA,
    objective_n_iter=RESULT_NB_ITERATIONS,
    objective_dataset_param_n_samples=NB_DATA_SAMPLES,
    objective_dataset_param_n_features=NB_DATA_FEATURES,
    objective_dataset_param_dtype=DTYPE,
    objective_dataset_param_random_state=DATA_RANDOM_STATE,
    objective_objective_param_n_clusters=NB_CLUSTERS,
    objective_objective_param_init=INIT_TYPE,
    objective_objective_param_random_state=SOLVER_RANDOM_STATE,
    objective_objective_param_sample_weight=DATA_SAMPLE_WEIGHTS,
    objective_solver_param___name=BACKEND_PROVIDER,
    objective_solver_param_device=COMPUTE_DEVICE,
    objective_solver_param_runtime=COMPUTE_RUNTIME,
    objective_solver_param_comment=COMMENT,
    objective_solver_param_version_info=VERSION_INFO,
    objective_solver_param_run_date=RUN_DATE,
    platform=PLATFORM,
)

PARQUET_TABLE_DISPLAY_MAPPING.update(
    {
        "platform-architecture": PLATFORM_ARCHITECTURE,
        "platform-release": PLATFORM_RELEASE,
        "system-cpus": SYSTEM_CPUS,
        "system-processor": SYSTEM_PROCESSOR,
        "system-ram (GB)": SYSTEM_RAM,
    }
)
_all_table_columns = list(PARQUET_TABLE_DISPLAY_MAPPING) + [BENCHMARK_ID_NAME]

ALL_EXPECTED_COLUMNS = set(BENCHMARK_DEFINING_COLUMNS + _all_table_columns)

IDS_LENGTH = 8


def _get_id_from_str(s):
    return hashlib.sha256(s.encode("utf8"), usedforsecurity=False).hexdigest()[
        :IDS_LENGTH
    ]


def _get_sample_id_for_columns(row, defining_colums, constant_identifier):
    return _get_id_from_str(
        "".join(row[defining_colums].astype(str)) + constant_identifier
    )


def _validate_one_parquet_table(source):
    df = pd.read_parquet(source)

    # NB: we're lenient on the columns
    for col in ALL_EXPECTED_COLUMNS - set(df.columns):
        df[col] = None

    df[BENCHMARK_ID_NAME] = df.apply(
        lambda row: _get_sample_id_for_columns(
            row, BENCHMARK_DEFINING_COLUMNS, _benchmark_defining_columns_identifier
        ),
        axis=1,
    )

    df = df[_all_table_columns]
    df.rename(columns=PARQUET_TABLE_DISPLAY_MAPPING, inplace=True, errors="raise")

    df[RUN_DATE] = df[RUN_DATE].astype("datetime64[ns]")

    return df


def _validate_one_csv_table(source, parse_dates=True, order_columns=True):
    NA_VALUES = set(STR_NA_VALUES)
    NA_VALUES.discard("None")

    df = pd.read_csv(
        source,
        usecols=TABLE_DISPLAY_ORDER,
        dtype=COLUMNS_DTYPES,
        index_col=False,
        na_values={col: NA_VALUES for col in COLUMNS_WITH_NONE_STRING},
        keep_default_na=False,
    )

    if order_columns:
        df = df[TABLE_DISPLAY_ORDER]

    if parse_dates:
        df[RUN_DATE] = pd.to_datetime(df[RUN_DATE], format=DATES_FORMAT).astype(
            "datetime64[ns]"
        )

    return df


def _assemble_output_table(
    dfs_from_csv, dfs_from_parquet, parquet_gpu_name, create_gpu_entry, list_known_gpus
):

    if not list_known_gpus and (len(dfs_from_parquet) == 0):
        if parquet_gpu_name is not None:
            parameter_name = (
                "--parquet-gpu-name" if parquet_gpu_name else "--no-parquet-gpu-name"
            )
            raise ValueError(
                f"The parameter {parameter_name} should only be used if at least one "
                "benchopt parquet table is being consolidated, but only got csv tables."
            )
        if create_gpu_entry is not False:
            raise ValueError(
                "The parameter --create-gpu-entry should only be used if at least one "
                "benchopt parquet table is being consolidated, but got only csv tables."
            )
    elif not list_known_gpus and parquet_gpu_name is None:
        raise ValueError(
            "Please use the --parquet-gpu-name parameter to provide a gpu name that "
            "will be added to the metadata of the samples in the input parquet tables "
            "or use the --no-parquet-gpu-name if you intend to leave the corresponding "
            "field empty."
        )

    else:
        gpu_names_from_csv = set(
            gpu_name
            for df in dfs_from_csv
            for gpu_name in df[SYSTEM_GPU]
            if (len(gpu_name) > 0)
        )

        if list_known_gpus:
            print("\n".join(gpu_names_from_csv))
            return False

        if (
            (len(parquet_gpu_name) > 0)
            and (parquet_gpu_name not in gpu_names_from_csv)
            and not create_gpu_entry
        ):
            raise IndexError(
                f"The gpu name {parquet_gpu_name} is unknown. Please use the "
                "--new-gpu-entry parameter to confirm the addition of the new gpu "
                "entry in the output csv table, or use --list-known-gpus parameter to "
                "print a list of gpus names that have been already registered and use "
                "one of those to bypass this error."
            )

        for df in dfs_from_parquet:
            df[SYSTEM_GPU] = parquet_gpu_name

    df_list = dfs_from_csv + dfs_from_parquet

    if len(df_list) > 1:
        df = pd.concat(df_list, ignore_index=True, copy=False)
    else:
        df = df_list[0]

    df = df[TABLE_DISPLAY_ORDER]
    df.sort_values(
        by=_row_sort_by, ascending=_row_sort_ascending, inplace=True, kind="stable"
    )
    # HACK: sanitize mix of None values and empty strings that can happen when some
    # columns are missing in the parquet input files (because it's optional and no
    # solver returns it in the batch) by passing the data to CSV and re-loading
    # again from CSV
    df = _sanitize_df_with_tocsv(df)

    df.drop_duplicates(subset=UNIQUE_BENCHMARK_KEY, inplace=True, ignore_index=True)

    return df


def _sanitize_df_with_tocsv(df):
    in_memory_buffer = BytesIO()
    _df_to_csv(df, in_memory_buffer)
    in_memory_buffer.seek(0)
    return _validate_one_csv_table(in_memory_buffer, order_columns=False)


def _df_to_csv(df, target):
    df.to_csv(target, index=False, mode="a", date_format=DATES_FORMAT)


def _gspread_sync(source, gspread_url, gspread_auth_key):
    import gspread

    df = _validate_one_csv_table(source, parse_dates=False)

    n_rows, n_cols = df.shape
    walltime_worksheet_col = df.columns.get_loc(WALLTIME) + 1

    gs = gspread.service_account(gspread_auth_key)
    sheet = gs.open_by_url(gspread_url)

    try:
        worksheet = sheet.worksheet(GOOGLE_WORKSHEET_NAME)
        worksheet.clear()
        worksheet.clear_basic_filter()
        worksheet.freeze(0, 0)
        worksheet.resize(rows=n_rows + 1, cols=n_cols)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(
            GOOGLE_WORKSHEET_NAME, rows=n_rows + 1, cols=n_cols
        )
        # ensure worksheets are sorted alphabetically
        sheet.reorder_worksheets(sorted(sheet.worksheets(), key=attrgetter("title")))

    # upload all values
    worksheet.update(
        values=[df.columns.values.tolist()] + df.values.tolist(), range_name="A1"
    )

    # set filter
    worksheet.set_basic_filter(1, 1, n_rows + 1, n_cols)

    # freeze filter rows and benchmark-defining cols
    worksheet.freeze(rows=1, cols=walltime_worksheet_col)

    format_queries = []

    # Text is centerd and wrapped in all cells
    global_format = dict(
        horizontalAlignment="CENTER",
        verticalAlignment="MIDDLE",
        wrapStrategy="WRAP",
    )
    global_range = (
        f"{gspread.utils.rowcol_to_a1(1, 1)}:"
        f"{gspread.utils.rowcol_to_a1(n_rows + 1, n_cols)}"
    )
    format_queries.append(dict(range=global_range, format=global_format))

    # benchmark_id and walltime columns are bold
    bold_format = dict(textFormat=dict(bold=True))
    benchmark_id_col_range = (
        f"{gspread.utils.rowcol_to_a1(2, 1)}:"
        f"{gspread.utils.rowcol_to_a1(n_rows + 1, 1)}"
    )
    walltime_col_range = (
        f"{gspread.utils.rowcol_to_a1(2, walltime_worksheet_col)}:"
        f"{gspread.utils.rowcol_to_a1(n_rows + 1, walltime_worksheet_col)}"
    )
    format_queries.append(dict(range=benchmark_id_col_range, format=bold_format))
    format_queries.append(dict(range=walltime_col_range, format=bold_format))

    # Header is light-ish yellow
    yellow_lighter_header = dict(
        backgroundColorStyle=dict(
            rgbColor=dict(red=1, green=1, blue=102 / 255, alpha=1)
        )
    )
    header_row_range = (
        f"{gspread.utils.rowcol_to_a1(1, 1)}:"
        f"{gspread.utils.rowcol_to_a1(1, n_cols)}"
    )
    format_queries.append(dict(range=header_row_range, format=yellow_lighter_header))

    # Every other benchmark_id has greyed background
    bright_gray_background = dict(
        backgroundColorStyle=dict(
            rgbColor=dict(red=232 / 255, green=233 / 255, blue=235 / 255, alpha=1)
        )
    )
    benchmark_ids = df[BENCHMARK_ID_NAME]
    benchmark_ids_ending_idx = (
        np.where((benchmark_ids.shift() != benchmark_ids).values[1:])[0] + 2
    )
    for benchmark_id_range_start, benchmark_id_range_end in zip_longest(
        *(iter(benchmark_ids_ending_idx),) * 2
    ):
        benchmark_row_range = (
            f"{gspread.utils.rowcol_to_a1(benchmark_id_range_start + 1, 1)}:"
            f"{gspread.utils.rowcol_to_a1(benchmark_id_range_end or (n_rows + 1), n_cols)}"  # noqa
        )
        format_queries.append(
            dict(range=benchmark_row_range, format=bright_gray_background)
        )

    # Apply formats
    worksheet.batch_format(format_queries)

    # auto-resize rows and cols
    worksheet.columns_auto_resize(0, n_cols - 1)
    worksheet.rows_auto_resize(0, n_rows)


if __name__ == "__main__":
    import os
    import sys
    from argparse import ArgumentParser

    argparser = ArgumentParser(
        description=(
            "Print an aggregated CSV-formated database of k-means benchmark results "
            "for the sklearn-engine-benchmarks project hosted at "
            "https://github.com/soda-inria/sklearn-engine-benchmarks.\n\n"
            "The inputs are assumed to be a collection of benchopt parquet files and "
            "CSV files, well formated according to the project current specs. This "
            "command assumes rhat the inputs are valid and is lenient at checking "
            "types, null values, or missing columns, hence the user is advised to "
            "cautiously check outputs before using.\n\n"
            "If several results are found for identical benchmarks, only the most "
            "recent `Run date` value is retained, all anterior entries are discarded "
            "from the output CSV."
        )
    )

    argparser.add_argument(
        "benchmark_files",
        nargs="+",
        help="benchopt parquet files or sklearn-engine-benchmarks csv files",
    )

    argparser.add_argument(
        "--check-csv",
        action="store_true",
        help="Perform a few sanity checks on a CSV database of k-means benchmark "
        "results. If this option is passed, then the command only expects a single "
        "input path to a csv file.",
    )

    argparser.add_argument(
        "--sync-to-gspread",
        action="store_true",
        help="Synchronize a CSV database of k-means benchmark results to a google "
        "spreadsheet and format it nicely. If this option is passed, then the command "
        "only expects a single input path to a csv file, and also requires "
        "--gspread-url and --gspread-auth-key.",
    )

    argparser.add_argument(
        "--gspread-url",
        help="URL to a google spreadsheet. Expected if and only if --sync-to-gspread "
        "is passed.",
    )

    argparser.add_argument(
        "--gspread-auth-key",
        help="Path to a json authentication key for a gspread service account. "
        "Expected if and only if --sync-to-gspread is passed.",
    )

    argparser.add_argument(
        "--parquet-gpu-name",
        help="Name of the GPU on the host that runs the benchmarks that are recorded "
        "in the input parquet files.",
    )

    argparser.add_argument(
        "--no-parquet-gpu-name",
        action="store_true",
        help="Do not insert a GPU name in the metadata of the benchmark samples that "
        "were recorded in the input parquet files (and leave it blank).",
    )

    argparser.add_argument(
        "--new-gpu-entry",
        action="store_true",
        help="Use this parameter along with --parquet-gpu-name to confirm that if the "
        "GPU name is not yet known in the existing databases, it will be added to the "
        "list of known GPU names. Else the command will throw an error.",
    )

    argparser.add_argument(
        "--list-known-gpus",
        action="store_true",
        help="Will print a list of the GPU names that are used in CSV benchmark files.",
    )

    args = argparser.parse_args()

    if (parquet_gpu_name := args.parquet_gpu_name) is None and args.no_parquet_gpu_name:
        parquet_gpu_name = ""

    create_gpu_entry = args.new_gpu_entry
    list_known_gpus = args.list_known_gpus

    paths = args.benchmark_files
    if (check_csv := args.check_csv) or args.sync_to_gspread:
        if (n_paths := len(paths)) > 1:
            command = "--check-csv" if check_csv else "--sync-to-gspread"
            raise ValueError(
                f"A single input path to a csv file is expected when the {command} "
                f"parameter is passed, but you passed {n_paths - 1} additional "
                "arguments."
            )
        path = paths[0]
        _, file_extension = os.path.splitext(path)
        if file_extension != ".csv":
            raise ValueError(
                "Expecting a '.csv' file extensions, but got "
                f"{file_extension} instead !"
            )

    if check_csv:
        df_loaded = _validate_one_csv_table(path)
        df_clean = _assemble_output_table(
            dfs_from_csv=[df_loaded],
            dfs_from_parquet=[],
            parquet_gpu_name=None,
            create_gpu_entry=False,
            list_known_gpus=list_known_gpus,
        )

        pd.testing.assert_frame_equal(df_loaded, df_clean)

    if gspread_sync := args.sync_to_gspread:
        if (gspread_url := args.gspread_url) is None:
            raise ValueError(
                "Please provide a URL to a google spreadsheet using the "
                "--gspread-url parameter."
            )

        if (gspread_auth_key := args.gspread_auth_key) is None:
            raise ValueError(
                "Please use the --gspread-auth-key parameter to pass a json "
                "authentication key for a service account from the google developer "
                "console."
            )
        _gspread_sync(path, gspread_url, gspread_auth_key)

    if not check_csv and not gspread_sync:
        dfs_from_parquet, dfs_from_csv = [], []
        for path in paths:
            _, file_extension = os.path.splitext(path)
            if file_extension == ".parquet":
                if list_known_gpus:
                    continue
                dfs_from_parquet.append(_validate_one_parquet_table(path))
            elif file_extension == ".csv":
                dfs_from_csv.append(_validate_one_csv_table(path, order_columns=False))
            else:
                raise ValueError(
                    "Expecting '.csv' or '.parquet' file extensions, but got "
                    f"{file_extension} instead !"
                )

        df = _assemble_output_table(
            dfs_from_csv=dfs_from_csv,
            dfs_from_parquet=dfs_from_parquet,
            parquet_gpu_name=parquet_gpu_name,
            create_gpu_entry=create_gpu_entry,
            list_known_gpus=list_known_gpus,
        )

        if df is not False:
            _df_to_csv(df, sys.stdout)
