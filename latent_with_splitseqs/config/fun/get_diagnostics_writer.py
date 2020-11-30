from self_supervised.utils.writer import MyWriterWithActivation
from self_sup_combined.base.writer.diagnostics_writer import \
    DiagnosticsWriter, DiagnosticsWriterBase


def get_diagnostics_writer(
        run_comment,
        config,
        scripts_to_copy,
        seed,
        config_path_name,
) -> DiagnosticsWriterBase:
    writer = MyWriterWithActivation(
        seed=seed,
        log_dir=config.log_folder,
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=config.log_interval,
        config=config,
        config_path_name=config_path_name,
        scripts_to_copy=scripts_to_copy,
    )
    return diagno_writer
