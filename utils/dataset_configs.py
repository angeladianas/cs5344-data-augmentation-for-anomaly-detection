"""Dataset-specific semantics for the shared tabular preprocessing pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """Declare feature semantics independently of observed storage dtypes."""

    name: str
    categorical: tuple[str, ...]
    counts: tuple[str, ...]
    continuous: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()
    bounded_rates: tuple[str, ...] = ()
    discrete: dict[str, list[int]] = field(default_factory=dict)
    label: str = "is_anomaly"
    plot_features: tuple[str, ...] = ()
    log_plot_features: tuple[str, ...] = ()

    @property
    def features(self):
        return [
            *self.categorical,
            *self.counts,
            *self.continuous,
            *self.binary,
            *self.bounded_rates,
            *self.discrete,
        ]

    def semantic_type(self, column):
        for kind, columns in [
            ("categorical", self.categorical),
            ("binary", self.binary),
            ("discrete_state", self.discrete),
            ("bounded_rate", self.bounded_rates),
            ("count", self.counts),
            ("continuous", self.continuous),
        ]:
            if column in columns:
                return kind
        raise ValueError(f"Unconfigured feature: {column}")


NSL_KDD = DatasetConfig(
    name="nsl_kdd",
    categorical=("protocol_type", "service", "flag"),
    binary=("land", "logged_in", "root_shell", "is_host_login", "is_guest_login"),
    discrete={"su_attempted": [0, 1, 2]},
    counts=(
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "count",
        "srv_count",
        "dst_host_count",
        "dst_host_srv_count",
    ),
    bounded_rates=(
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ),
    plot_features=("src_bytes", "dst_bytes", "duration", "count", "serror_rate", "same_srv_rate"),
    log_plot_features=("src_bytes", "dst_bytes", "duration"),
)

UNSW_NB15 = DatasetConfig(
    name="unsw_nb15",
    categorical=("proto", "service", "state"),
    binary=("is_sm_ips_ports",),
    # Released rows contain 0, 1, 2, and 4: preserve them instead of forcing binary.
    discrete={"is_ftp_login": [0, 1, 2, 4]},
    continuous=(
        "dur",
        "rate",
        "sload",
        "dload",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "tcprtt",
        "synack",
        "ackdat",
    ),
    counts=(
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "swin",
        "stcpb",
        "dtcpb",
        "dwin",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
    ),
    plot_features=("sbytes", "dbytes", "dur", "rate", "spkts", "tcprtt"),
    log_plot_features=("sbytes", "dbytes", "dur", "rate"),
)
