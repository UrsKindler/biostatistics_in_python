import rpy2.robjects as ro

ro.r(
    """
    install.packages(
        "VennDiagram",
        repos = "https://cloud.r-project.org"
    )
    """
)

ro.r("library(VennDiagram)")
print("VennDiagram wurde erfolgreich installiert und geladen.")