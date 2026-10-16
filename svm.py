from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluacion_clasificadores import (
	ROOT,
	cargar_y_dividir,
	evaluar_y_guardar,
	guardar_comparacion,
)

SUMMARY_FILE = ROOT / "svm_comparacion_resultados.csv"
C_VALUES = (0.1, 1, 10, 100)
GAMMA_VALUES = ("scale", "auto")


def entrenar_y_evaluar() -> list[dict[str, object]]:
	particion = cargar_y_dividir()
	configuraciones: list[dict[str, object]] = []
	for c_value in C_VALUES:
		configuraciones.append({"kernel": "linear", "C": c_value})
		for gamma in GAMMA_VALUES:
			configuraciones.append({"kernel": "rbf", "C": c_value, "gamma": gamma})

	resultados: list[dict[str, object]] = []
	for configuracion in configuraciones:
		kernel = str(configuracion["kernel"])
		c_slug = str(configuracion["C"]).replace(".", "_")
		gamma = configuracion.get("gamma")
		gamma_slug = f"_gamma_{gamma}" if gamma is not None else ""
		carpeta = ROOT / f"svm_{kernel}_c_{c_slug}{gamma_slug}_resultados"
		modelo = make_pipeline(
			StandardScaler(),
			SVC(**configuracion, class_weight="balanced", random_state=42),
		)
		print(f"\nConfiguración SVM: {configuracion}")
		resultados.append(
			evaluar_y_guardar(
				modelo=modelo,
				particion=particion,
				tipo_modelo="SVM",
				parametros=configuracion,
				carpeta_salida=carpeta,
			)
		)

	guardar_comparacion(resultados, SUMMARY_FILE)
	return resultados


if __name__ == "__main__":
	entrenar_y_evaluar()
