from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "Dataset_limpio4_preprocesado" / "caracteristicas.csv"
SUMMARY_FILE = ROOT / "knn_comparacion_resultados.csv"

RANDOM_STATE = 42
FOLDS = 5
K_VALUES = (1, 3, 5, 7, 9, 11, 13, 15)


def entrenar_y_evaluar() -> dict[str, object]:
	if not DATASET_FILE.is_file():
		raise FileNotFoundError(f"No se encontró el CSV de características: {DATASET_FILE}")

	dataframe = pd.read_csv(DATASET_FILE)
	columnas_requeridas = {"clase", "archivo_origen"}
	faltantes = columnas_requeridas - set(dataframe.columns)
	if faltantes:
		raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(faltantes)}")

	columnas_metadata = {"clase", "archivo", "archivo_origen", "segmento"}
	columnas_caracteristicas = [
		columna for columna in dataframe.columns if columna not in columnas_metadata
	]
	if not columnas_caracteristicas:
		raise ValueError("El CSV no contiene columnas de características para entrenar.")

	caracteristicas = dataframe[columnas_caracteristicas].apply(pd.to_numeric, errors="coerce")
	if caracteristicas.isna().any().any():
		columnas_invalidas = caracteristicas.columns[caracteristicas.isna().any()].tolist()
		raise ValueError(f"Hay valores vacíos o no numéricos en: {columnas_invalidas}")

	etiquetas = dataframe["clase"].astype(str)
	grupos = dataframe["archivo_origen"].astype(str)
	clases = sorted(etiquetas.unique())
	if len(clases) < 2:
		raise ValueError("Se necesitan al menos dos clases para entrenar el modelo.")

	divisor = StratifiedGroupKFold(
		n_splits=FOLDS,
		shuffle=True,
		random_state=RANDOM_STATE,
	)
	indices_entrenamiento, indices_prueba = next(
		divisor.split(caracteristicas, etiquetas, groups=grupos)
	)
	x_entrenamiento = caracteristicas.iloc[indices_entrenamiento]
	x_prueba = caracteristicas.iloc[indices_prueba]
	y_entrenamiento = etiquetas.iloc[indices_entrenamiento]
	y_prueba = etiquetas.iloc[indices_prueba]
	grupos_entrenamiento = set(grupos.iloc[indices_entrenamiento])
	grupos_prueba = set(grupos.iloc[indices_prueba])

	if grupos_entrenamiento & grupos_prueba:
		raise RuntimeError("El split comparte audios de origen entre entrenamiento y prueba.")
	if set(y_entrenamiento) != set(clases) or set(y_prueba) != set(clases):
		raise ValueError("El split debe contener todas las clases en entrenamiento y prueba.")

	if max(K_VALUES) > len(y_entrenamiento):
		raise ValueError("Algún valor de K supera el número de muestras de entrenamiento.")

	resultados_corridas: list[dict[str, object]] = []
	for k in K_VALUES:
		output_dir = ROOT / f"knn_{k}_resultados"
		output_dir.mkdir(parents=True, exist_ok=True)
		model_file = output_dir / "modelo_knn.joblib"
		metrics_file = output_dir / "metricas.json"
		confusion_csv = output_dir / "matriz_confusion.csv"
		confusion_png = output_dir / "matriz_confusion.png"

		modelo = make_pipeline(
			StandardScaler(),
			KNeighborsClassifier(n_neighbors=k, weights="distance"),
		)
		modelo.fit(x_entrenamiento, y_entrenamiento)
		predicciones = modelo.predict(x_prueba)

		accuracy = float(accuracy_score(y_prueba, predicciones))
		precision, recall, f1_score, _ = precision_recall_fscore_support(
			y_prueba,
			predicciones,
			labels=clases,
			average="macro",
			zero_division=0,
		)
		matriz = confusion_matrix(y_prueba, predicciones, labels=clases)
		reporte_clases = classification_report(
			y_prueba,
			predicciones,
			labels=clases,
			output_dict=True,
			zero_division=0,
		)

		joblib.dump(
			{
				"modelo": modelo,
				"k": k,
				"caracteristicas": columnas_caracteristicas,
				"clases": clases,
				"semilla": RANDOM_STATE,
			},
			model_file,
		)

		matriz_dataframe = pd.DataFrame(matriz, index=clases, columns=clases)
		matriz_dataframe.index.name = "real"
		matriz_dataframe.to_csv(confusion_csv, encoding="utf-8")

		figura, eje = plt.subplots(figsize=(7, 6))
		imagen = eje.imshow(matriz, cmap="Blues")
		eje.set_xticks(range(len(clases)), clases, rotation=35, ha="right")
		eje.set_yticks(range(len(clases)), clases)
		eje.set_xlabel("Predicción")
		eje.set_ylabel("Etiqueta real")
		eje.set_title(f"Matriz de confusión KNN (K={k})")
		for fila in range(len(clases)):
			for columna in range(len(clases)):
				eje.text(columna, fila, str(matriz[fila, columna]), ha="center", va="center")
		figura.colorbar(imagen, ax=eje, label="Muestras")
		figura.tight_layout()
		figura.savefig(confusion_png, dpi=160)
		plt.close(figura)

		resultados: dict[str, object] = {
			"modelo": "KNeighborsClassifier",
			"k": k,
			"accuracy": accuracy,
			"precision_macro": float(precision),
			"recall_macro": float(recall),
			"f1_score_macro": float(f1_score),
			"muestras_entrenamiento": int(len(indices_entrenamiento)),
			"muestras_prueba": int(len(indices_prueba)),
			"audios_entrenamiento": len(grupos_entrenamiento),
			"audios_prueba": len(grupos_prueba),
			"caracteristicas": columnas_caracteristicas,
			"clases": clases,
			"reporte_por_clase": reporte_clases,
		}
		metrics_file.write_text(
			json.dumps(resultados, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		resultados_corridas.append(
			{
				"k": k,
				"accuracy": accuracy,
				"f1_score_macro": float(f1_score),
				"recall_macro": float(recall),
				"precision_macro": float(precision),
				"carpeta": output_dir.name,
			}
		)

		print(f"\nK={k}")
		print(f"Accuracy:  {accuracy:.4f}")
		print(f"F1-score:  {f1_score:.4f} (macro)")
		print(f"Recall:    {recall:.4f} (macro)")
		print(f"Precision: {precision:.4f} (macro)")
		print("Matriz de confusión (filas reales, columnas predichas):")
		print(matriz_dataframe.to_string())
		print(f"Artefactos guardados en: {output_dir}")

	comparacion = pd.DataFrame(resultados_corridas)
	comparacion.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
	print("\nComparación por K:")
	print(comparacion.drop(columns="carpeta").to_string(index=False))
	print(f"Resumen guardado en: {SUMMARY_FILE}")
	return {"corridas": resultados_corridas, "resumen": str(SUMMARY_FILE)}


if __name__ == "__main__":
	entrenar_y_evaluar()
