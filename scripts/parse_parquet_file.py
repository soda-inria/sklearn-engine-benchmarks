import pandas as pd
import hashlib


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
RESULT_NB_ITERATIONS = "Result nb iterations"
RESULT_INERTIA = "Result inertia"
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
    RESULT_NB_ITERATIONS,
    RESULT_INERTIA,
    PLATFORM,
    DATA_RANDOM_STATE,
    SOLVER_RANDOM_STATE,
    RUN_DATE,
    COMMENT,
]

# Importance and say if ascending / descending
ROW_SORT_ORDER = [
    (DTYPE, True),
    (NB_DATA_SAMPLES, False),
    (NB_DATA_FEATURES, False),
    (NB_CLUSTERS, False),
    (INIT_TYPE, True),
    (DATA_SAMPLE_WEIGHTS, True),
    (WALLTIME, True),
    (BACKEND_PROVIDER, True),
    (COMPUTE_DEVICE, True),
    (COMPUTE_RUNTIME, True),
    (RESULT_NB_ITERATIONS, True),
    (RESULT_INERTIA, False),
    (PLATFORM, True),
    (DATA_RANDOM_STATE, True),
    (SOLVER_RANDOM_STATE, True),
    (RUN_DATE, True),
    (COMMENT, True),
    (BENCHMARK_ID_NAME, True),
]
_row_sort_by, _row_sort_ascending = map(list, zip(*ROW_SORT_ORDER))

TABLE_DISPLAY_MAPPING = dict(
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
    objective_solver_param_run_date=RUN_DATE,
    platform=PLATFORM,
)

TABLE_DISPLAY_MAPPING.update(
    {
        "platform-architecture": "Platform architecture",
        "platform-release": "Platform release",
        "system-cpus": "Nb cpus",
        "system-processor": "Cpu name",
        "system-ram (GB)": "RAM (GB)",
    }
)
_all_table_columns = list(TABLE_DISPLAY_MAPPING) + [BENCHMARK_ID_NAME]

ALL_EXPECTED_COLUMNS = set(BENCHMARK_DEFINING_COLUMNS + _all_table_columns)

IDS_LENGTH = 8


def get_id_from_str(s):
    return hashlib.sha256(s.encode("utf8"), usedforsecurity=False).hexdigest()[
        :IDS_LENGTH
    ]


def get_sample_id_for_columns(row, defining_colums, constant_identifier):
    return get_id_from_str(
        "".join(row[defining_colums].astype(str)) + constant_identifier
    )


PARQUET_DUMP_PATH = "/root/sklearn-engine-benchmarks/benchmarks/kmeans/outputs/benchopt_run_2023-09-15_08h16m08.parquet"  # noqa
df = pd.read_parquet(PARQUET_DUMP_PATH)

for col in ALL_EXPECTED_COLUMNS - set(df.columns):
    df[col] = None

df[BENCHMARK_ID_NAME] = df.apply(
    lambda row: get_sample_id_for_columns(
        row, BENCHMARK_DEFINING_COLUMNS, _benchmark_defining_columns_identifier
    ),
    axis=1,
)

df = df[_all_table_columns]
df.rename(columns=TABLE_DISPLAY_MAPPING, inplace=True, errors="raise")
df = df[TABLE_DISPLAY_ORDER]
df.sort_values(
    by=_row_sort_by, ascending=_row_sort_ascending, inplace=True, kind="stable"
)


# TODO: add comment
# TODO: add display date
