# Benchmark del TFG

Módulo independiente para evaluar, de forma reproducible, la calidad de
los artefactos que genera el sistema (**quiz** y **presentación PPTX**)
con varios modelos Ollama y varios PDFs.

El módulo **no depende de Streamlit**: se ejecuta por script y genera
artefactos, métricas automáticas y prompts de evaluación externa para
ChatGPT / Gemini.

## 1. Objetivo

Justificar experimentalmente en la memoria / tribunal que el sistema:

1. Funciona sobre PDFs de dominios distintos.
2. Produce quiz y PPTX con calidad medible.
3. Mantiene un comportamiento razonable con modelos locales pequeños
   (3B–14B), no solo con los grandes.

## 2. Estructura

```
benchmark/
├── __init__.py
├── README.md              ← este fichero
├── config.py              ← modelos por defecto, umbrales, rutas
├── runner.py              ← CLI principal
├── metrics.py             ← métricas automáticas (quiz + pptx)
├── judge_prompts.py       ← prompts de evaluación externa con IA
├── reports.py             ← agregación en CSV
├── dataset/
│   ├── pdfs/              ← PDFs del benchmark (opcional; fallback: ../PDF)
│   └── catalog.json       ← metadatos del dataset, editable a mano
├── results/               ← una carpeta por ejecución (pdf × modelo)
└── reports/               ← CSVs agregados y rúbrica
```

## 3. Cómo se ejecuta

Desde la raíz del proyecto (con el venv activo):

```powershell
# Todos los PDFs con todos los modelos por defecto
python -m benchmark.runner

# Subconjunto de modelos y PDFs
python -m benchmark.runner --models qwen2.5:7b mistral:7b --pdfs poo bbdd

# Solo quiz, o solo PPTX
python -m benchmark.runner --only-quiz
python -m benchmark.runner --only-pptx

# Ver qué se ejecutaría, sin ejecutar
python -m benchmark.runner --dry-run
```

El runner reutiliza el pipeline de `src/` sin tocar el core:

```
PDF  -->  pdf_processor.process_pdf
      -->  map_reduce.build_knowledge_base       (una vez por ejecución)
      -->  quiz_generator.generate_quiz           (si --only-pptx no está)
      -->  pptx_generator.generate_presentation   (si --only-quiz no está)
```

## 4. Outputs por ejecución

En `benchmark/results/<pdf_id>__<modelo_sanitizado>/`:

| Fichero                       | Contenido                                    |
| ----------------------------- | -------------------------------------------- |
| `metrics.json`                | Métricas automáticas + timings + KB info     |
| `quiz.json`                   | Quiz generado (si aplica)                    |
| `presentation.pptx`           | Presentación generada (si aplica)            |
| `plan.json`                   | Plan de la presentación (estructura+bullets) |
| `kb.json`                     | KnowledgeBase que sirvió como fuente         |
| `eval_prompts/quiz_eval.txt`  | Prompt de evaluación del quiz con IA externa |
| `eval_prompts/pptx_eval.txt`  | Prompt de evaluación del PPTX con IA externa |
| `error.log`                   | Traza si algo falló (no rompe el benchmark)  |

Reports agregados en `benchmark/reports/`:

* `benchmark_summary.csv` — histórico acumulado con política **upsert**
  por clave (`pdf_id`, `model`):
  - si ejecutas una combinación nueva, añade fila;
  - si repites el mismo `pdf_id` y el mismo `model`, reemplaza su fila.
  Incluye además trazas de longitud del documento (`pdf_num_chars_catalog`,
  `pdf_num_chars_extracted`, `pdf_truncated`). Es el fichero principal
  para análisis.
* `model_averages.csv` — media por modelo de las métricas numéricas,
  calculada sobre todo el histórico `status=ok`.
* `manual_evaluation_template.csv` — plantilla para puntuar a mano una
  muestra pequeña (rúbrica 1-5, mismos criterios que la IA externa),
  también basada en el histórico acumulado.
* `rubric_reference.txt` — texto con la rúbrica completa.

## 5. Tres niveles de evaluación

El benchmark se apoya en tres capas de evaluación, **complementarias**:

1. **Automática** — `metrics.py`. Recuentos y solapamientos simples.
   Objetivos: detectar fallos duros (quiz vacío, bullets cortados,
   repetición entre slides, opciones desbalanceadas). Definida por un
   `score_quiz` y `score_pptx` en `[0, 1]` con pesos documentados.

2. **Semiautomática (IA externa)** — `judge_prompts.py`. Cada ejecución
   produce dos `.txt` listos para pegar en ChatGPT / Gemini adjuntando
   el PDF original. El evaluador devuelve un JSON con **rúbrica 1-5**
   por criterio. El investigador recopila esos JSON (manualmente o con
   un script posterior).

3. **Manual** — `manual_evaluation_template.csv`. Plantilla para
   puntuar con la misma rúbrica una muestra pequeña (p. ej. 3-5
   ejecuciones representativas). Útil como validación cualitativa.

## 6. ¿Qué mide exactamente cada métrica automática?

### Quiz

| Métrica                      | Qué mide                                               |
| ---------------------------- | ------------------------------------------------------ |
| `num_questions`              | Preguntas devueltas                                    |
| `bloom_distribution`         | Cuenta por nivel Bloom declarado                       |
| `bloom_diversity`            | Nº de niveles Bloom distintos cubiertos                |
| `duplicate_pairs`            | Pares de preguntas con stems muy similares (Jaccard)   |
| `unbalanced_options_count`   | Preguntas cuyo `max/min` de longitudes de opciones es ≥ 2.2 |
| `banned_phrases_count`       | Apariciones de "todas las anteriores", etc.            |
| `pct_with_explanation`       | Fracción con justificación útil (≥10 chars)            |
| `kb_term_coverage`           | Fracción de términos clave de la KB mencionados en el quiz |
| `score_quiz`                 | Media ponderada 0..1 (ver `rubric_reference.txt`)      |

### PPTX

| Métrica                       | Qué mide                                                 |
| ----------------------------- | -------------------------------------------------------- |
| `num_slides_total`            | Portada + índice + contenido + conclusión                |
| `num_content_slides`          | Diapositivas de desarrollo con bullets                   |
| `avg_bullets_per_slide`       | Media de bullets por slide de contenido                  |
| `bullets_too_long/short`      | Bullets fuera del rango razonable                        |
| `bullets_possibly_truncated`  | Bullets que no cierran en puntuación natural             |
| `cross_slide_repetition_pairs`| Pares de bullets muy parecidos entre slides distintas    |
| `index_coherence`             | Fracción de títulos del índice que aparecen en slides    |
| `score_pptx`                  | Media ponderada 0..1                                     |

> Todas las métricas son deliberadamente simples. No pretenden
> sustituir al juicio humano ni a la evaluación por IA externa, pero
> detectan patológicos de forma objetiva y barata.

## 7. Dataset

`dataset/catalog.json` describe los PDFs. Campos:

```jsonc
{
  "id": "poo",                          // identificador corto
  "filename": "ProgOrientadaObjetos.pdf",
  "title": "Programación Orientada a Objetos",
  "num_chars": 26380,                   // longitud aproximada del texto extraído
  "has_tables": false,
  "has_images": true,
  "domain": "programacion",
  "length_category": "medio"            // bajo (0-20k), medio (20k-50k), alto (>50k)
}
```

Criterio recomendado para `length_category` (por `num_chars`):

- `bajo`: 0 a 20.000
- `medio`: 20.001 a 50.000
- `alto`: más de 50.000

Los PDFs pueden estar en `benchmark/dataset/pdfs/` **o** en la carpeta
`PDF/` de la raíz del proyecto: el runner los busca en ese orden.

Para añadir un PDF nuevo: editar `catalog.json` y colocar el fichero en
una de las dos carpetas. No hace falta tocar código.

## 8. Robustez

* Si una ejecución falla (Ollama caído, modelo no instalado, error del
  pipeline…), el runner **lo anota** (`error.log` + fila con `status=error`
  en el CSV) y **sigue** con la siguiente combinación.
* Los fallos internos entre quiz y PPTX están aislados: un quiz que
  falla no impide generar el PPTX.
* Los imports pesados (`src.quiz_generator`, etc.) están diferidos: 
  `python -m benchmark.runner --help` funciona aunque Ollama no esté
  disponible.

## 9. Limitaciones conocidas

* Las métricas automáticas son heurísticas: un `score_quiz` alto no
  garantiza calidad pedagógica, solo ausencia de patologías obvias. De
  ahí la necesidad de la capa de IA externa y el muestreo manual.
* La métrica de cobertura (`kb_term_coverage`) es por solapamiento
  léxico, no semántico. Puede infravalorar quizzes con buena
  paráfrasis.
* El runner usa **una ejecución por combinación**. La variabilidad
  entre corridas (temperatura, stochasticity) no se mide aún; ampliar
  con repeticiones es trivial (bucle externo) si fuera necesario.
* El evaluador externo (ChatGPT / Gemini) puede rechazar el prompt si
  no se adjunta el PDF; en tal caso el propio prompt pide evaluación
  cautelosa y menciona la limitación en `summary_comment`.

## 10. Flujo de trabajo recomendado para el TFG

1. Ejecutar el benchmark con 2-3 modelos representativos
   (p. ej. `qwen2.5:14b`, `qwen2.5:7b`, `llama3.2:3b`) sobre los 5 PDFs
   del catálogo.
2. Revisar `benchmark_summary.csv` y `model_averages.csv`.
3. Para ejecuciones dudosas o interesantes, pegar `eval_prompts/*.txt`
   en ChatGPT / Gemini con el PDF adjunto y guardar el JSON devuelto.
4. Rellenar `manual_evaluation_template.csv` con una muestra pequeña.
5. Resumir conclusiones en la memoria citando los tres niveles.
