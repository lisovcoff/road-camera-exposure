# Development / Разработка

[English home](../README.md) · [Главная](../README.ru.md)

Tests need no hardware. / Для тестов оборудование не нужно.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
roadcam simulate
roadcam run --config configs/module3.example.json
```

Tests cover policy, conversion, timing/gating, configuration, budgets and mocked camera lifecycle. Mock success is not hardware success. CI targets Python 3.10/3.13. / Проверяются алгоритм, преобразования, время/команды, конфигурация, лимиты и имитация камеры. Это не подтверждение реального запуска.

Optional charts / Графики:

```sh
python -m pip install -e '.[analysis]'
python tools/render_results.py --input assets/results/measurements.json --output runs/charts-001
```

Charts use public aggregates, not raw sensor data. Output must be new. / Графики строятся по агрегатам, не по исходникам сенсора. Нужен новый каталог.

```sh
roadcam analyze-motion --help
roadcam analyze-static --help
python -m pip wheel --no-deps --wheel-dir dist .
```

Analyzers read legacy capture layouts encoded in their loaders: `trial_*`, setup/summary/observations and `.npy` frames. They do not automatically process arbitrary MP4 or the new live validation directory. Full historical inputs are not distributed. / Анализаторы читают прежнюю структуру записи из своих загрузчиков. Они не обрабатывают произвольные MP4 и новый каталог живой проверки автоматически. Полные исторические исходники не распространяются.

Keep SDK imports in adapters. Unit, sensor-mode and clock changes require tests and actual readback validation. Contributions should record hardware, versions, configuration and rejected frames. No private captures, credentials or site-specific network settings. / SDK остаётся в адаптере. Изменения единиц, режимов и часов требуют тестов и реальных метаданных. Для измерений указывайте оборудование, версии, конфигурацию и исключённые кадры. Не публикуйте частные записи, пароли и локальные сетевые настройки.
