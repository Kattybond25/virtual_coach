# **Виртуальный тренер**

## **Описание проекта**

Этот проект представляет собой систему для автоматического анализа и оценки правильности выполнения физических упражнений. Система использует компьютерное зрение и машинное обучение для обнаружения ключевых точек человеческой позы и сравнения их с идеальными образцами. Основная цель — помочь пользователям улучшить технику выполнения упражнений, предоставляя обратную связь на основе анализа видео.

---

## **Основные функции**

1. **Обнаружение позы**:
   - Используется библиотека MediaPipe для обнаружения ключевых точек человеческой позы.

2. **Сравнение поз**:
   - Реализован алгоритм Procrustes Analysis для сравнения двух наборов ключевых точек.
   - Улучшенная версия Procrustes Analysis с вращением и масштабированием для более точного совмещения.

3. **Визуализация**:
   - Визуализация сравнения идеального и пользовательского видео с отрисовкой ключевых точек и скелета.

4. **Графический интерфейс**:
   - Пользовательский интерфейс на основе Tkinter для загрузки видео и запуска анализа.

5. **Оценка точности**:
   - Вычисление процента сходства между пользовательским и идеальным выполнением упражнений.

---

## **Технический стек**

- **Язык программирования**: Python 3.x
- **Библиотеки**:
  - OpenCV: Обработка видео и изображений.
  - MediaPipe: Обученная модель для обнаружения ключевых точек позы.
  - NumPy: Работа с массивами и вычислениями.
  - SciPy: Реализация Procrustes Analysis и оптимизации.
  - Tkinter: Создание графического интерфейса.
  - Threading: Параллельное выполнение задач.

---

## **Установка**

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/Kattybond25/virtual_coach.git
   cd virtual_coach
   ```

2. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

   Файл `requirements.txt` должен содержать следующие библиотеки:
   ```
   opencv-python
   mediapipe
   numpy
   scipy
   tkinter
   ```

3. **Запустите программу**:
   ```bash
   python coach.py
   ```

---

## **Использование**

1. **Загрузите видео**:
   - Выберите идеальное видео упражнения из предоставленного списка.
   - Загрузите свое видео с выполнением упражнения.

2. **Запустите анализ**:
   - Нажмите кнопку "Сравнить", чтобы начать анализ.

3. **Получите результаты**:
   - Система выведет оценку сходства (от 0 до 100%) и визуализирует сравнение видео.

---
