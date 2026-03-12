from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHORE_JAR = PROJECT_ROOT / "Chore.jar"
DB_TABLE_OPTIONS = (
    "tap_response_data",
    "tap_baseline_data",
    "tstat_gene_data",
    "tstat_allele_data",
    "gene_MSD",
    "allele_MSD",
    "psa_summarised_data",
)

SCREEN_OPTIONS = [
    "PD_Screen",
    "ASD_Screen",
    "G-Proteins_Screen",
    "Glia_Genes_Screen",
    "Neuron_Genes_Screen",
    "PD_GWAS_Locus22_Screen",
    "PD_GWAS_Locus71_Screen",
    "ASD_WGS_Screen",
    "Miscellaneous",
]


@dataclass(slots=True)
class PipelineConfig:
    raw_data_dir: Path
    screen: str
    output_dir: Path
    chore_jar: Path = DEFAULT_CHORE_JAR
    run_chore: bool = True
    upload_to_db: bool = False
    database_ini: Path | None = None
    db_upload_mode: str = "append"
    replace_tables: tuple[str, ...] = ()
    number_of_taps: int = 30
    isi_seconds: int = 10
    first_tap_seconds: int = 600
    java_heap: str = "13g"

    @property
    def recovery_tap(self) -> int:
        return self.number_of_taps + 1
