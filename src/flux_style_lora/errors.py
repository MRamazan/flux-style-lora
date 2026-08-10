class FluxStyleLoraError(RuntimeError):
    pass


class ConfigurationError(FluxStyleLoraError):
    pass


class EnvironmentPreparationError(FluxStyleLoraError):
    pass


class AssetError(FluxStyleLoraError):
    pass


class VaeValidationError(AssetError):
    pass


class DatasetError(FluxStyleLoraError):
    pass


class TrainingError(FluxStyleLoraError):
    pass


class CheckpointError(FluxStyleLoraError):
    pass


class EvaluationError(FluxStyleLoraError):
    pass


class ExportError(FluxStyleLoraError):
    pass


class BundleValidationError(FluxStyleLoraError):
    pass


class BundleImportError(FluxStyleLoraError):
    pass


class UnsupportedBundleVersionError(BundleValidationError):
    pass


class BundleIntegrityError(BundleValidationError):
    pass
