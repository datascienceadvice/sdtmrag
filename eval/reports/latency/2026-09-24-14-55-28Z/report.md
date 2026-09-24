# Оценка качества RAG (sdtmrag)

- Дата прогона: 2026-09-24 14:55:28Z
- Вопросов в наборе: 71 (eval/gold.en.jsonl)
- Индекс: `data/index_clean_tuned.sqlite=clean_tuned`
- k: [5]

## Сводка

| Конфигурация | Модель эмбеддингов | Реранк | Чанков | MRR | Задержка p50, мс |
|---|---|---|---:|---:|---:|
| clean_tuned | bge-small-en-v1.5 | bge-reranker-base | 4111 | 0.567 | 24873 |

### Precision@k

| k | clean_tuned |
|---|---:|
| 5 | 0.200 |

### Recall@k

| k | clean_tuned |
|---|---:|
| 5 | 1.000 |

### nDCG@k

| k | clean_tuned |
|---|---:|
| 5 | 0.673 |

### Hit@k

| k | clean_tuned |
|---|---:|
| 5 | 1.000 |

### Задержка (мс)

| Конфигурация | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| clean_tuned | 25209 | 24873 | 26325 | 26325 |

## Разбивка по языку вопроса (Recall@5)

| Язык | clean_tuned |
|---|---:|
| en | 1.000 (n=3) |

## Разбивка по темам (Recall@5)

| Тема | clean_tuned |
|---|---:|
| LB | 1.000 (n=3) |

## MRR по темам

| Тема | clean_tuned |
|---|---:|
| LB | 0.567 |

_Столбцы во всех таблицах: clean_tuned._
