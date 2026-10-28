# Оценка качества RAG (sdtmrag)

- Дата прогона: 2026-09-24 13:41:24Z
- Вопросов в наборе: 48 (eval/gold.judge.jsonl)
- Индекс: `data/index_clean_tuned.sqlite=clean+tuned`
- k: [1, 5, 10]

## Сводка

| Конфигурация | Модель эмбеддингов | Реранк | Чанков | MRR | Задержка p50, мс |
|---|---|---|---:|---:|---:|
| clean+tuned | bge-small-en-v1.5 | bge-reranker-base | 4111 | 0.646 | 42546 |

### Precision@k

| k | clean+tuned |
|---|---:|
| 1 | 0.500 |
| 5 | 0.273 |
| 10 | 0.191 |

### Recall@k

| k | clean+tuned |
|---|---:|
| 1 | 0.500 |
| 5 | 0.864 |
| 10 | 0.864 |

### nDCG@k

| k | clean+tuned |
|---|---:|
| 1 | 0.500 |
| 5 | 0.701 |
| 10 | 0.701 |

### Hit@k

| k | clean+tuned |
|---|---:|
| 1 | 0.500 |
| 5 | 0.864 |
| 10 | 0.864 |

### Задержка (мс)

| Конфигурация | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| clean+tuned | 41516 | 42546 | 47211 | 50615 |

### Качество ответов (LLM-судья, 0..1)

| Конфигурация | Faithfulness | Citation accuracy | Correctness | Верных отказов |
|---|---:|---:|---:|---:|
| clean+tuned | 0.999 | 0.998 | 0.979 | 4/4 |

## Разбивка по языку вопроса (Recall@5)

| Язык | clean+tuned |
|---|---:|
| en | 0.939 (n=33) |
| ru | 0.636 (n=11) |

## Разбивка по темам (Recall@5)

| Тема | clean+tuned |
|---|---:|
| LB | 0.944 (n=18) |
| datasets | 0.800 (n=5) |
| general-model | 0.889 (n=9) |
| pdf-sections | 0.750 (n=12) |

## MRR по темам

| Тема | clean+tuned |
|---|---:|
| LB | 0.724 |
| datasets | 0.600 |
| general-model | 0.452 |
| pdf-sections | 0.694 |

_Столбцы во всех таблицах: clean+tuned._
