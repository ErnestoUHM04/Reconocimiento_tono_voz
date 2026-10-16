from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "Dataset_limpio4_preprocesado" / "caracteristicas.csv"
RANDOM_STATE = 42
FOLDS = 5


@dataclass
class DatosParticion:
	columnas_caracteristicas: list[str]
	clases: list[str]
	x_entrenamiento: pd.DataFrame
	x_prueba: pd.DataFrame
	y_entrenamiento: pd.Series
	y_prueba: pd.Series
	grupos_entrenamiento: set[str]
	grupos_prueba: set[str]


def cargar_y_dividir() -> DatosParticion:
	if not DATASET_FILE.is_file():
		raise FileNotFoundError(f"No se encontró el CSV de características: {DATASET_FILE}")

	dataframe = pd.read_csv(DATASET_FILE)
	columnas_requeridas = {"clase", "archivo_origen"}
	faltantes = columnas_requeridas - set(dataframe.columns)
	if faltantes:
		raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(faltantes)}")
	if dataframe["clase"].isna().any() or dataframe["archivo_origen"].isna().any():
		raise ValueError("El CSV contiene etiquetas o archivos de origen vacíos.")

	columnas_metadata = {"clase", "archivo", "archivo_origen", "segmento"}
	columnas_caracteristicas = [
		columna for columna in dataframe.columns if columna not in columnas_metadata
	]
	if not columnas_caracteristicas:
		raise ValueError("El CSV no contiene características para entrenar.")

	caracteristicas = dataframe[columnas_caracteristicas].apply(pd.to_numeric, errors="coerce")
	valores = caracteristicas.to_numpy(dtype=np.float64)
	if not np.isfinite(valores).all():
		columnas_invalidas = caracteristicas.columns[
			~np.isfinite(valores).all(axis=0)
		].tolist()
		raise ValueError(f"Hay valores vacíos o no finitos en: {columnas_invalidas}")

	etiquetas = dataframe["clase"].astype(str)
	grupos = dataframe["archivo_origen"].astype(str)
	clases = sorted(etiquetas.unique())
	if len(clases) < 2:
		raise ValueError("Se necesitan al menos dos clases para entrenar.")

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
		raise RuntimeError("El split comparte audios entre entrenamiento y prueba.")
	if set(y_entrenamiento) != set(clases) or set(y_prueba) != set(clases):
		raise ValueError("El split debe contener todas las clases en ambos conjuntos.")

	return DatosParticion(
		columnas_caracteristicas=columnas_caracteristicas,
		clases=clases,
		x_entrenamiento=x_entrenamiento,
		x_prueba=x_prueba,
		y_entrenamiento=y_entrenamiento,
		y_prueba=y_prueba,
		grupos_entrenamiento=grupos_entrenamiento,
		grupos_prueba=grupos_prueba,
	)


def evaluar_y_guardar(
	modelo: BaseEstimator,
	particion: DatosParticion,
	tipo_modelo: str,
	parametros: dict[str, object],
	carpeta_salida: Path,
) -> dict[str, object]:
	modelo.fit(particion.x_entrenamiento, particion.y_entrenamiento)
	predicciones = modelo.predict(particion.x_prueba)
	accuracy = float(accuracy_score(particion.y_prueba, predicciones))
	precision, recall, f1_score, _ = precision_recall_fscore_support(
		particion.y_prueba,
		predicciones,
		labels=particion.clases,
		average="macro",
		zero_division=0,
	)
	matriz = confusion_matrix(
		particion.y_prueba,
		predicciones,
		labels=particion.clases,
	)
	reporte_clases = classification_report(
		particion.y_prueba,
		predicciones,
		labels=particion.clases,
		output_dict=True,
		zero_division=0,
	)

	carpeta_salida.mkdir(parents=True, exist_ok=True)
	joblib.dump(
		{
			"modelo": modelo,
			"tipo_modelo": tipo_modelo,
			"parametros": parametros,
			"caracteristicas": particion.columnas_caracteristicas,
			"clases": particion.clases,
			"semilla": RANDOM_STATE,
		},
		carpeta_salida / "modelo.joblib",
	)

	matriz_dataframe = pd.DataFrame(
		matriz,
		index=particion.clases,
		columns=particion.clases,
	)
	matriz_dataframe.index.name = "real"
	matriz_dataframe.to_csv(carpeta_salida / "matriz_confusion.csv", encoding="utf-8")

	figura, eje = plt.subplots(figsize=(7, 6))
	imagen = eje.imshow(matriz, cmap="Blues")
	eje.set_xticks(range(len(particion.clases)), particion.clases, rotation=35, ha="right")
	eje.set_yticks(range(len(particion.clases)), particion.clases)
	eje.set_xlabel("Predicción")
	eje.set_ylabel("Etiqueta real")
	eje.set_title(f"Matriz de confusión {tipo_modelo}")
	for fila in range(len(particion.clases)):
		for columna in range(len(particion.clases)):
			eje.text(columna, fila, str(matriz[fila, columna]), ha="center", va="center")
	figura.colorbar(imagen, ax=eje, label="Muestras")
	figura.tight_layout()
	figura.savefig(carpeta_salida / "matriz_confusion.png", dpi=160)
	plt.close(figura)

	resultados: dict[str, object] = {
		"modelo": tipo_modelo,
		"parametros": parametros,
		"accuracy": accuracy,
		"precision_macro": float(precision),
		"recall_macro": float(recall),
		"f1_score_macro": float(f1_score),
		"muestras_entrenamiento": len(particion.y_entrenamiento),
		"muestras_prueba": len(particion.y_prueba),
		"audios_entrenamiento": len(particion.grupos_entrenamiento),
		"audios_prueba": len(particion.grupos_prueba),
		"caracteristicas": particion.columnas_caracteristicas,
		"clases": particion.clases,
		"reporte_por_clase": reporte_clases,
	}
	(carpeta_salida / "metricas.json").write_text(
		json.dumps(resultados, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	print(f"Accuracy:  {accuracy:.4f}")
	print(f"F1-score:  {f1_score:.4f} (macro)")
	print(f"Recall:    {recall:.4f} (macro)")
	print(f"Precision: {precision:.4f} (macro)")
	print("Matriz de confusión (filas reales, columnas predichas):")
	print(matriz_dataframe.to_string())
	print(f"Artefactos guardados en: {carpeta_salida}")

	return {
		"accuracy": accuracy,
		"f1_score_macro": float(f1_score),
		"recall_macro": float(recall),
		"precision_macro": float(precision),
		"carpeta": carpeta_salida.name,
		**parametros,
	}


def guardar_comparacion(resultados: list[dict[str, object]], ruta: Path) -> None:
	comparacion = pd.DataFrame(resultados)
	comparacion.to_csv(ruta, index=False, encoding="utf-8")
	print("\nComparación de configuraciones:")
	print(comparacion.to_string(index=False))
	print(f"Resumen guardado en: {ruta}")