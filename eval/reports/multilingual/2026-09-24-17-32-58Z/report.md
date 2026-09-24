# Оценка качества RAG (sdtmrag)

- Дата прогона: 2026-09-24 17:32:58Z
- Вопросов в наборе: 96 (eval/gold.en.jsonl, eval/gold.ru.jsonl)
- Индекс: `data/index_clean.sqlite=clean-bge-small, data/index_clean_e5.sqlite=clean-e5-small`
- k: [1, 3, 5, 10]

## Сводка

| Конфигурация | Модель эмбеддингов | Реранк | Чанков | MRR | Задержка p50, мс |
|---|---|---|---:|---:|---:|
| clean-bge-small | bge-small-en-v1.5 | выключен | 3418 | 0.584 | 21 |
| clean-e5-small | multilingual-e5-small | выключен | 3418 | 0.655 | 17 |

### Precision@k

| k | clean-bge-small | clean-e5-small |
|---|---:|---:|
| 1 | 0.461 | 0.528 |
| 3 | 0.281 | 0.285 |
| 5 | 0.225 | 0.211 |
| 10 | 0.155 | 0.133 |

### Recall@k

| k | clean-bge-small | clean-e5-small |
|---|---:|---:|
| 1 | 0.461 | 0.528 |
| 3 | 0.674 | 0.742 |
| 5 | 0.775 | 0.843 |
| 10 | 0.854 | 0.888 |

### nDCG@k

| k | clean-bge-small | clean-e5-small |
|---|---:|---:|
| 1 | 0.461 | 0.528 |
| 3 | 0.584 | 0.655 |
| 5 | 0.624 | 0.698 |
| 10 | 0.649 | 0.712 |

### Hit@k

| k | clean-bge-small | clean-e5-small |
|---|---:|---:|
| 1 | 0.461 | 0.528 |
| 3 | 0.674 | 0.742 |
| 5 | 0.775 | 0.843 |
| 10 | 0.854 | 0.888 |

### Задержка (мс)

| Конфигурация | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| clean-bge-small | 24 | 21 | 37 | 51 |
| clean-e5-small | 17 | 17 | 21 | 24 |

## Разбивка по языку вопроса (Recall@5)

| Язык | clean-bge-small | clean-e5-small |
|---|---:|---:|
| en | 0.879 (n=66) | 0.894 (n=66) |
| ru | 0.478 (n=23) | 0.696 (n=23) |

## Разбивка по темам (Recall@5)

| Тема | clean-bge-small | clean-e5-small |
|---|---:|---:|
| LB | 0.778 (n=36) | 1.000 (n=36) |
| datasets | 0.636 (n=11) | 0.727 (n=11) |
| general-model | 0.824 (n=17) | 0.765 (n=17) |
| pdf-sections | 0.800 (n=25) | 0.720 (n=25) |

## MRR по темам

| Тема | clean-bge-small | clean-e5-small |
|---|---:|---:|
| LB | 0.545 | 0.848 |
| datasets | 0.564 | 0.654 |
| general-model | 0.548 | 0.451 |
| pdf-sections | 0.674 | 0.517 |

_Столбцы во всех таблицах: clean-bge-small, clean-e5-small._
