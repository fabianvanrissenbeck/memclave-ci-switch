# Building the CI-switch

The CI-switch build depends only on the fsl image. This is embedded
in the CI-switch binary at link time. Simply place the fsl image
in the root of the CI-switch directory and then compile everything
via cmake.