from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from Callback import ReproductorAudio
from jj import RANGOS_VOCALES, analizar_tipo_voz

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
OUTPUT = ROOT / "Dataset_limpio"
MANIFEST = ROOT / "manifesto_voz.csv"
RESULTS_FILE = ROOT / "resultados_evaluacion.json"
REPORT_FILE = ROOT / "reporte_evaluacion.txt"
SAMPLES_PER_TYPE = 20

ALIASES = {
    "Bajo": {"bajo", "bass", "bassus", "bassus ch1-f4", "bassus ch2-f4"},
    "Baritono": {"baritono", "baritone", "bari"},
    "Tenor": {"tenor", "tenor i", "tenor ii", "tenor ch1-c4", "tenor ch2-c4"},
    "Contralto": {"contralto", "alto"},
    "Mezzosoprano": {"mezzosoprano"},
    "Soprano": {"soprano", "soprano "},
}


def normalizar_etiqueta(nombre: str) -> str | None:
    """Obtiene la etiqueta de voz desde el nombre del archivo WAV."""
    etiqueta = nombre.strip().lower()
    for tipo, etiquetas in ALIASES.items():
        if etiqueta in etiquetas:
            return tipo
    return None


def descubrir_archivos() -> dict[str, list[Path]]:
    """Busca WAV de voz y usa el tag de info.json, con respaldo en el nombre."""
    grupos: dict[str, list[Path]] = {tipo: [] for tipo in RANGOS_VOCALES}
    for carpeta in sorted(DATASET.iterdir()):
        if not carpeta.is_dir():
            continue
        voces_dir = carpeta / "voices"
        if not voces_dir.is_dir():
            continue

        etiquetas_por_archivo: dict[str, str] = {}
        info_file = carpeta / "info.json"
        if info_file.is_file():
            try:
                parts = json.loads(info_file.read_text(encoding="utf-8"))
                for part in parts.get("parts", []):
                    nombre_archivo = str(part.get("name", "")).strip().casefold()
                    tipo_original = str(part.get("type", "")).strip().casefold()
                    etiqueta = normalizar_etiqueta(tipo_original)
                    if etiqueta is None:
                        etiqueta = normalizar_etiqueta(nombre_archivo)
                    if etiqueta is not None:
                        etiquetas_por_archivo[nombre_archivo] = etiqueta
            except (OSError, json.JSONDecodeError, TypeError):
                etiquetas_por_archivo = {}

        for archivo in sorted(voces_dir.glob("*.wav")):
            etiqueta = normalizar_etiqueta(archivo.stem)
            if etiqueta is None:
                etiqueta = etiquetas_por_archivo.get(archivo.stem.strip().casefold())
            if etiqueta is not None:
                grupos[etiqueta].append(archivo)
    return grupos


def seleccionar_muestras(grupos: dict[str, list[Path]]) -> list[tuple[str, Path]]:
    """Selecciona 20 archivos deterministas de cada clase disponible."""
    tipos_disponibles = [tipo for tipo in RANGOS_VOCALES if grupos[tipo]]
    seleccionadas: list[tuple[str, Path]] = []
    for tipo in tipos_disponibles:
        if len(grupos[tipo]) < SAMPLES_PER_TYPE:
            raise ValueError(
                f"Faltan {SAMPLES_PER_TYPE - len(grupos[tipo])} muestras de {tipo}; "
                f"se encontraron {len(grupos[tipo])}."
            )
        seleccionadas.extend((tipo, path) for path in grupos[tipo][:SAMPLES_PER_TYPE])
    return seleccionadas


def tipos_disponibles(grupos: dict[str, list[Path]]) -> list[str]:
    """Devuelve las clases con al menos un WAV disponible."""
    return [tipo for tipo in RANGOS_VOCALES if grupos[tipo]]


def limpiar_output() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)


def preparar_dataset(seleccion: list[tuple[str, Path]]) -> list[dict[str, object]]:
    """Copia los WAV con nombres TipoVocal_#.wav y crea el manifiesto."""
    limpiar_output()
    registros: list[dict[str, object]] = []
    for numero, (tipo, origen) in enumerate(seleccion, start=1):
        destino = OUTPUT / f"{tipo}_{numero:03d}.wav"
        shutil.copy2(origen, destino)
        audio, frecuencia = sf.read(destino, dtype="float32")
        if audio.size == 0 or frecuencia <= 0:
            raise ValueError(f"El archivo {destino.name} no contiene audio válido.")
        reproductor = ReproductorAudio(str(destino))
        if reproductor.audio.size != audio.size or reproductor.frecuencia != frecuencia:
            raise ValueError(f"La lectura de {destino.name} no es consistente.")
        registros.append(
            {
                "numero": numero,
                "nombre": destino.name,
                "tipo_esperado": tipo,
                "origen": str(origen.relative_to(DATASET)),
                "frecuencia_hz": int(frecuencia),
                "samples": int(audio.size),
            }
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=["numero", "nombre", "tipo_esperado", "origen", "frecuencia_hz", "samples"],
        )
        writer.writeheader()
        writer.writerows(registros)
    return registros


def analizar_evaluaciones(registros: list[dict[str, object]]) -> list[dict[str, object]]:
    """Evalúa cada WAV con la función de clasificación de jj.py."""
    resultados: list[dict[str, object]] = []
    ventana_segundos = 12.0
    for registro in registros:
        ruta = OUTPUT / str(registro["nombre"])
        audio, frecuencia = sf.read(ruta, dtype="float32")
        audio_evaluacion = audio[: int(ventana_segundos * frecuencia)]
        print(f"Analizando {registro['nombre']} ({registro['tipo_esperado']})")
        resultado = analizar_tipo_voz(audio_evaluacion, frecuencia)
        if "error" in resultado:
            predicted = "ERROR"
            confidence = 0.0
        else:
            predicted = resultado["tipo"]
            scores = resultado["puntuaciones"]
            confidence = float(max(scores.values(), key=lambda item: item["porcentaje_f0"])["porcentaje_f0"])
        resultados.append(
            {
                "numero": registro["numero"],
                "nombre": registro["nombre"],
                "tipo_esperado": registro["tipo_esperado"],
                "tipo_predicho": predicted,
                "confidencia": confidence,
                "f0_min_hz": resultado.get("f0_min"),
                "f0_max_hz": resultado.get("f0_max"),
                "f0_mediana_hz": resultado.get("f0_mediana"),
                "error": resultado.get("error"),
            }
        )
    return resultados


def escribir_resultados(resultados: list[dict[str, object]]) -> None:
    with RESULTS_FILE.open("w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, ensure_ascii=False, indent=2)


def crear_matriz(
    resultados: list[dict[str, object]],
    tipos: list[str],
) -> tuple[dict[str, object], list[list[int]]]:
    matriz = [[0 for _ in tipos] for _ in tipos]
    for resultado in resultados:
        esperado = str(resultado["tipo_esperado"])
        predicho = str(resultado["tipo_predicho"])
        if esperado not in tipos or predicho not in tipos:
            continue
        matriz[tipos.index(esperado)][tipos.index(predicho)] += 1

    correctos = sum(matriz[i][i] for i in range(len(tipos)))
    total = len(resultados)
    metrics = {
        "total_evaluados": total,
        "correctos": correctos,
        "accuracy": correctos / total if total else 0.0,
        "porcentaje_correctos": 100.0 * correctos / total if total else 0.0,
        "errores": [
            {
                "nombre": resultado["nombre"],
                "esperado": resultado["tipo_esperado"],
                "predicho": resultado["tipo_predicho"],
            }
            for resultado in resultados
            if resultado["tipo_predicho"] != resultado["tipo_esperado"]
        ],
    }
    return metrics, matriz


def mostrar_matriz(matriz: list[list[int]], tipos: list[str], metrics: dict[str, object]) -> None:
    fig, eje = plt.subplots(figsize=(9, 7))
    image = eje.imshow(matriz, cmap="Blues", vmin=0, vmax=max(1, max(max(row) for row in matriz)))
    eje.set_xticks(range(len(tipos)), tipos, rotation=45, ha="right")
    eje.set_yticks(range(len(tipos)), tipos)
    eje.set_xlabel("Tipo predicho")
    eje.set_ylabel("Tipo esperado")
    eje.set_title("Matriz de confusión de clasificación vocal")

    for fila in range(len(tipos)):
        for columna in range(len(tipos)):
            valor = matriz[fila][columna]
            eje.text(columna, fila, str(valor), ha="center", va="center", color="black")

    totales_por_fila = np.sum(matriz, axis=1)
    for indice, valor in enumerate(totales_por_fila):
        eje.set_yticklabels(tipos, rotation=0)
        if valor:
            eje.text(-1.05, indice, str(valor), ha="right", va="center")

    fig.colorbar(image, ax=eje, label="Número de archivos")
    fig.text(0.5, 0.02, f"Accuracy: {metrics['accuracy']:.3f}", ha="center", va="top")
    fig.tight_layout()
    fig.savefig(ROOT / "matriz_confusion.png", dpi=180)
    plt.close(fig)


def escribir_reporte(
    resultados: list[dict[str, object]],
    metrics: dict[str, object],
    tipos: list[str],
) -> None:
    errores = metrics["errores"]
    lines = [
        "REPORTE DE EVALUACIÓN DE VOICES",
        "==============================",
        f"Total evaluados: {metrics['total_evaluados']}",
        f"Correctos: {metrics['correctos']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Porcentaje correcto: {metrics['porcentaje_correctos']:.2f}%",
        "",
        "CLASIFICACIÓN POR TIPO",
    ]
    for tipo in tipos:
        count = sum(1 for item in resultados if item["tipo_esperado"] == tipo)
        correct = sum(1 for item in resultados if item["tipo_esperado"] == tipo and item["tipo_predicho"] == tipo)
        lines.append(f"{tipo}: {correct}/{count} correctos")
    lines.extend(["", "ERRORES DE CLASIFICACIÓN:"])
    if errores:
        for error in errores:
            lines.append(f"- {error['nombre']}: esperado {error['esperado']}, predicho {error['predicho']}")
    else:
        lines.append("- Sin errores.")
    lines.extend(
        [
            "",
            "RANGO DE F0 USADO POR EL CLASIFICADOR:",
            *[f"{tipo}: {rango[0]}–{rango[1]} Hz" for tipo, rango in RANGOS_VOCALES.items() if tipo in tipos],
            "",
            "NOTA: Los resultados se basan en el nombre del archivo, mientras que la",
            "clasificación se obtiene con el analizador MFCC/F0 de jj.py.",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara 20 WAV por tipo de voz y ejecuta la evaluación de jj.py."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Comprueba disponibilidad sin copiar ni evaluar archivos.",
    )
    args = parser.parse_args()

    grupos = descubrir_archivos()
    tipos = tipos_disponibles(grupos)
    counts = {tipo: len(grupos[tipo]) for tipo in tipos}
    print("Distribución detectada:")
    for tipo in tipos:
        print(f"  {tipo}: {counts[tipo]} archivos")

    try:
        seleccion = seleccionar_muestras(grupos)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 2

    if args.dry_run:
        print(f"Se seleccionarían {len(seleccion)} archivos ({SAMPLES_PER_TYPE} por tipo disponible).")
        return 0

    registros = preparar_dataset(seleccion)
    resultados = analizar_evaluaciones(registros)
    escribir_resultados(resultados)
    metrics, matriz = crear_matriz(resultados, tipos)
    mostrar_matriz(matriz, tipos, metrics)
    escribir_reporte(resultados, metrics, tipos)

    print(f"Se copiaron {len(registros)} archivos a {OUTPUT.relative_to(ROOT)}")
    print(f"Se evaluaron {len(resultados)} archivos.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Matriz: {ROOT / 'matriz_confusion.png'}")
    print(f"Resultados: {RESULTS_FILE}")
    print(f"Reporte: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
