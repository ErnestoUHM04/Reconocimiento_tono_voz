from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from evaluacion_clasificadores import (
	ROOT,
	cargar_y_dividir,
	evaluar_y_guardar,
	guardar_comparacion,
)

SUMMARY_FILE = ROOT / "randomforest_comparacion_resultados.csv"
N_ESTIMATORS_VALUES = (100, 200, 300)
MAX_DEPTH_VALUES: tuple[int | None, ...] = (None, 10, 20)


def entrenar_y_evaluar() -> list[dict[str, object]]:
	particion = cargar_y_dividir()
	resultados: list[dict[str, object]] = []

	for n_estimators in N_ESTIMATORS_VALUES:
		for max_depth in MAX_DEPTH_VALUES:
			configuracion: dict[str, object] = {
				"n_estimators": n_estimators,
				"max_depth": max_depth,
				"max_features": "sqrt",
				"class_weight": "balanced",
				"random_state": 42,
				"n_jobs": -1,
			}
			depth_slug = str(max_depth) if max_depth is not None else "sin_limite"
			carpeta = ROOT / (
				f"randomforest_{n_estimators}_arboles_"
				f"profundidad_{depth_slug}_resultados"
			)
			modelo = RandomForestClassifier(**configuracion)
			print(f"\nConfiguración Random Forest: {configuracion}")
			resultados.append(
				evaluar_y_guardar(
					modelo=modelo,
					particion=particion,
					tipo_modelo="RandomForest",
					parametros=configuracion,
					carpeta_salida=carpeta,
				)
			)

	guardar_comparacion(resultados, SUMMARY_FILE)
	return resultados


if __name__ == "__main__":
	entrenar_y_evaluar()
