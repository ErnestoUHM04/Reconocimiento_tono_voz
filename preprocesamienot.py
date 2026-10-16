from __future__ import annotations

import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset_limpio"
OUTPUT = ROOT / "Dataset_limpio4_preprocesado"
STAGING = OUTPUT / ".staging"
SEGMENTS_OUTPUT = OUTPUT / "audios_segmentados"
FEATURES_CSV = OUTPUT / "caracteristicas.csv"

SEGMENT_SECONDS = 30
MAX_SEGMENTS_PER_AUDIO = 3
SILENCE_THRESHOLD_DB = 30
MFCC_COUNT = 13


def cargar_sin_silencios(path: Path) -> tuple[np.ndarray, int]:
	audio, sample_rate = sf.read(path, dtype="float32")
	if audio.ndim > 1:
		audio = audio.mean(axis=1)
	audio = np.nan_to_num(audio)
	intervals = librosa.effects.split(audio, top_db=SILENCE_THRESHOLD_DB)
	if len(intervals) == 0:
		return np.array([], dtype=np.float32), sample_rate
	return np.concatenate([audio[start:end] for start, end in intervals]), sample_rate


def resumen(valores: np.ndarray) -> tuple[float, float]:
	valores = np.asarray(valores, dtype=np.float64)
	valores = valores[np.isfinite(valores) & (valores > 0)]
	if valores.size == 0:
		return 0.0, 0.0
	return float(np.mean(valores)), float(np.std(valores))


def resumen_finito(valores: np.ndarray) -> tuple[float, float]:
	valores = np.asarray(valores, dtype=np.float64)
	valores = valores[np.isfinite(valores)]
	if valores.size == 0:
		return 0.0, 0.0
	return float(np.mean(valores)), float(np.std(valores))


def extraer_caracteristicas(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
	features: dict[str, float] = {}
	mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=MFCC_COUNT)
	for indice, coeficientes in enumerate(mfcc, start=1):
		features[f"MFCC_{indice:02d}_media"] = float(np.mean(coeficientes))
		features[f"MFCC_{indice:02d}_desv"] = float(np.std(coeficientes))

	audio_centrado = audio - np.mean(audio)
	zms = float(np.mean(np.square(audio_centrado)))
	features["ZMS"] = zms
	features["ZMS_dB"] = float(10 * np.log10(max(zms, np.finfo(float).tiny)))
	features["RMS"] = float(np.sqrt(zms))

	timbre = {
		"centroide_Hz": librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
		"ancho_Hz": librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
		"rolloff_Hz": librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
		"flatness": librosa.feature.spectral_flatness(y=audio),
	}
	for nombre, valores in timbre.items():
		media, desviacion = resumen_finito(valores)
		features[f"Timbre_{nombre}_media"] = media
		features[f"Timbre_{nombre}_desv"] = desviacion
	contraste = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
	for banda, valores in enumerate(contraste, start=1):
		media, desviacion = resumen_finito(valores)
		features[f"Timbre_contraste_banda_{banda}_media"] = media
		features[f"Timbre_contraste_banda_{banda}_desv"] = desviacion

	sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
	pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
	f0_values = pitch.selected_array["frequency"]
	voiced_f0 = f0_values[np.isfinite(f0_values) & (f0_values > 0)]
	f0_mean, f0_std = resumen(voiced_f0)
	features["F0_media"] = f0_mean
	features["F0_desv"] = f0_std
	if voiced_f0.size:
		altura_semitonos = librosa.hz_to_midi(voiced_f0)
		features["Altura_media_semitonos"] = float(np.mean(altura_semitonos))
		features["Altura_mediana_semitonos"] = float(np.median(altura_semitonos))
		features["Altura_min_Hz"] = float(np.min(voiced_f0))
		features["Altura_max_Hz"] = float(np.max(voiced_f0))
		features["Altura_rango_Hz"] = float(np.ptp(voiced_f0))
	else:
		features.update(
			{
				"Altura_media_semitonos": 0.0,
				"Altura_mediana_semitonos": 0.0,
				"Altura_min_Hz": 0.0,
				"Altura_max_Hz": 0.0,
				"Altura_rango_Hz": 0.0,
			}
		)

	formants = sound.to_formant_burg(
		time_step=0.01,
		max_number_of_formants=5,
		maximum_formant=5500,
		window_length=0.025,
		pre_emphasis_from=50,
	)
	times = np.arange(0.02, len(audio) / sample_rate - 0.02, 0.02)
	for numero in range(1, 4):
		valores = np.array([formants.get_value_at_time(numero, time) for time in times])
		media, desviacion = resumen(valores)
		features[f"F{numero}_media"] = media
		features[f"F{numero}_desv"] = desviacion

	point_process = call(sound, "To PointProcess (periodic, cc)", 60, 500)
	jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
	shimmer = call(
		[sound, point_process],
		"Get shimmer (local)",
		0,
		0,
		0.0001,
		0.02,
		1.3,
		1.6,
	)
	features["Jitter"] = float(jitter) if np.isfinite(jitter) else 0.0
	features["Shimmer"] = float(shimmer) if np.isfinite(shimmer) else 0.0
	harmonicity = sound.to_harmonicity_cc(
		time_step=0.01,
		minimum_pitch=60,
		silence_threshold=0.1,
		periods_per_window=1.0,
	)
	hnr_mean = call(harmonicity, "Get mean", 0, 0)
	hnr_std = call(harmonicity, "Get standard deviation", 0, 0)
	features["HNR_media_dB"] = float(hnr_mean) if np.isfinite(hnr_mean) else 0.0
	features["HNR_desv_dB"] = float(hnr_std) if np.isfinite(hnr_std) else 0.0
	return features


def procesar_dataset() -> pd.DataFrame:
	archivos = sorted(DATASET.glob("*.wav"))
	if not archivos:
		raise FileNotFoundError(f"No se encontraron archivos WAV en {DATASET}.")

	clases_origen = {path.stem.rsplit("_", 1)[0] for path in archivos}
	if STAGING.exists():
		shutil.rmtree(STAGING)
	STAGING.mkdir(parents=True)

	candidatos: dict[str, list[dict[str, object]]] = {clase: [] for clase in clases_origen}
	for path in archivos:
		clase = path.stem.rsplit("_", 1)[0]
		try:
			audio, sample_rate = cargar_sin_silencios(path)
			tamano_segmento = SEGMENT_SECONDS * sample_rate
			for indice in range(MAX_SEGMENTS_PER_AUDIO):
				inicio = indice * tamano_segmento
				fin = inicio + tamano_segmento
				if fin > len(audio):
					break
				segmento = audio[inicio:fin]
				features = extraer_caracteristicas(segmento, sample_rate)
				nombre = f"{path.stem}_segmento_{indice + 1:02d}.wav"
				temporal = STAGING / nombre
				sf.write(temporal, segmento, sample_rate)
				candidatos[clase].append(
					{
						"path": temporal,
						"source": path.name,
						"segment": indice + 1,
						"features": features,
					}
				)
		except (OSError, ValueError, RuntimeError) as error:
			print(f"Se omite {path.name}: {error}")

	sin_muestras = [clase for clase, muestras in candidatos.items() if not muestras]
	if sin_muestras:
		raise ValueError(f"No quedaron segmentos válidos para estas clases: {sin_muestras}")

	cantidad_balanceada = min(len(muestras) for muestras in candidatos.values())
	if SEGMENTS_OUTPUT.exists():
		shutil.rmtree(SEGMENTS_OUTPUT)
	SEGMENTS_OUTPUT.mkdir(parents=True)

	registros: list[dict[str, object]] = []
	for clase in sorted(candidatos):
		for muestra in candidatos[clase][:cantidad_balanceada]:
			destino = SEGMENTS_OUTPUT / Path(muestra["path"]).name
			shutil.move(str(muestra["path"]), destino)
			registros.append(
				{
					"clase": clase,
					"archivo": str(destino.relative_to(OUTPUT)),
					"archivo_origen": muestra["source"],
					"segmento": muestra["segment"],
					**muestra["features"],
				}
			)

	shutil.rmtree(STAGING)
	dataframe = pd.DataFrame(registros)
	conteos = dataframe["clase"].value_counts().to_dict()
	if len(set(conteos.values())) != 1:
		raise RuntimeError(f"El conjunto quedó desbalanceado: {conteos}")
	OUTPUT.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(FEATURES_CSV, index=False, encoding="utf-8")
	print(f"Muestras por clase: {conteos}")
	print(f"CSV: {FEATURES_CSV}")
	print(f"Audios segmentados: {SEGMENTS_OUTPUT}")
	return dataframe


if __name__ == "__main__":
	procesar_dataset()