import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  # Для стилей отрисовки

# Список идеальных видео
IDEAL_VIDEOS = {
    "Упражнение 1": "1.mp4",
    "Упражнение 2": "2.mp4",
    "Упражнение 3": "3.mp4",
    "Упражнение 4": "4.mp4",
    "Упражнение 5": "5.mp4"
}

# Функция для получения разрешения экрана
def get_screen_resolution():
    """
    Получает разрешение экрана.
    :return: Кортеж (ширина, высота) экрана.
    """
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

# Функция для отображения видео по центру
def show_video_centered(frame, window_name='Pose Detection'):
    """
    Отображает кадр по центру экрана.
    :param frame: Кадр для отображения.
    :param window_name: Имя окна.
    """
    # Получаем разрешение экрана
    screen_width, screen_height = get_screen_resolution()

    # Получаем размеры кадра
    frame_height, frame_width = frame.shape[:2]

    # Устанавливаем размер окна
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)

    # Рассчитываем позицию окна для центрирования
    x = (screen_width - frame_width) // 2
    y = (screen_height - frame_height) // 2

    # Устанавливаем позицию окна
    cv2.moveWindow(window_name, x, y)

    # Отображаем кадр
    cv2.imshow(window_name, frame)

# Функция для масштабирования видео до 1280x720
def resize_video(video_path, target_width=1280, target_height=720):
    """
    Масштабирует видео до указанных размеров.

    :param video_path: Путь к исходному видео.
    :param target_width: Целевая ширина видео.
    :param target_height: Целевая высота видео.
    :return: Путь к масштабированному видео.
    """
    # Генерируем новое имя для масштабированного видео
    base_name = video_path.split('/')[-1].split('.')[0]  # Имя файла без расширения
    resized_video_path = f"{base_name}_resized.mp4"

    # Открываем исходное видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    # Получаем параметры исходного видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Создаём объект для записи масштабированного видео
    out = cv2.VideoWriter(resized_video_path, fourcc, fps, (target_width, target_height))

    # Масштабируем каждый кадр
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)

    # Освобождаем ресурсы
    cap.release()
    out.release()

    return resized_video_path

# Функция для обнаружения ключевых точек на видео
def detect_pose_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])  # Убираем координату z
            keypoints_list.append(keypoints)

    cap.release()
    return np.array(keypoints_list)

# Функция для сравнения двух поз с использованием Procrustes Analysis
def procrustes_analysis(pose1, pose2):
    # Проверка размерности массивов
    if pose1.ndim == 1:
        pose1 = pose1[np.newaxis, :]  # Добавляем второе измерение
    if pose2.ndim == 1:
        pose2 = pose2[np.newaxis, :]  # Добавляем второе измерение

    # Центрирование и нормализация
    pose1 = pose1 - np.mean(pose1, axis=0)
    pose2 = pose2 - np.mean(pose2, axis=0)
    pose1 /= np.std(pose1)
    pose2 /= np.std(pose2)

    # Вычисление матрицы расстояний
    cost_matrix = cdist(pose1, pose2, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_distance = cost_matrix[row_ind, col_ind].sum()
    return total_distance

# Функция для визуализации сравнения видео покадрово
def visualize_comparison(ideal_keypoints, user_keypoints, ideal_video_path, user_video_path):
    """
    Визуализирует сравнение видео покадрово.
    :param ideal_keypoints: Ключевые точки идеального видео.
    :param user_keypoints: Ключевые точки пользовательского видео.
    :param ideal_video_path: Путь к идеальному видео.
    :param user_video_path: Путь к пользовательскому видео.
    """
    ideal_cap = cv2.VideoCapture(ideal_video_path)
    user_cap = cv2.VideoCapture(user_video_path)

    # Получаем FPS видео
    fps = int(ideal_cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / fps)  # Задержка между кадрами в миллисекундах

    frame_idx = 0

    while ideal_cap.isOpened() and user_cap.isOpened():
        ret_ideal, ideal_frame = ideal_cap.read()
        ret_user, user_frame = user_cap.read()

        if not ret_ideal or not ret_user:
            break

        # Рисуем ключевые точки идеального видео на кадре пользовательского видео (зеленый цвет)
        if frame_idx < len(ideal_keypoints):
            for (x, y) in ideal_keypoints[frame_idx]:
                x, y = int(x * user_frame.shape[1]), int(y * user_frame.shape[0])
                cv2.circle(user_frame, (x, y), 5, (0, 255, 0), -1)  # Зеленые точки

        # Рисуем ключевые точки пользовательского видео на кадре пользовательского видео (красный цвет)
        if frame_idx < len(user_keypoints):
            for (x, y) in user_keypoints[frame_idx]:
                x, y = int(x * user_frame.shape[1]), int(y * user_frame.shape[0])
                cv2.circle(user_frame, (x, y), 5, (0, 0, 255), -1)  # Красные точки

        # Рисуем скелет идеального видео (зеленый цвет)
        if frame_idx < len(ideal_keypoints):
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(ideal_keypoints[frame_idx]) and end_idx < len(ideal_keypoints[frame_idx]):
                    start_point = (int(ideal_keypoints[frame_idx][start_idx][0] * user_frame.shape[1]),
                                   int(ideal_keypoints[frame_idx][start_idx][1] * user_frame.shape[0]))
                    end_point = (int(ideal_keypoints[frame_idx][end_idx][0] * user_frame.shape[1]),
                                 int(ideal_keypoints[frame_idx][end_idx][1] * user_frame.shape[0]))
                    cv2.line(user_frame, start_point, end_point, (0, 255, 0), 2)  # Зеленые линии

        # Рисуем скелет пользовательского видео (красный цвет)
        if frame_idx < len(user_keypoints):
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(user_keypoints[frame_idx]) and end_idx < len(user_keypoints[frame_idx]):
                    start_point = (int(user_keypoints[frame_idx][start_idx][0] * user_frame.shape[1]),
                                   int(user_keypoints[frame_idx][start_idx][1] * user_frame.shape[0]))
                    end_point = (int(user_keypoints[frame_idx][end_idx][0] * user_frame.shape[1]),
                                 int(user_keypoints[frame_idx][end_idx][1] * user_frame.shape[0]))
                    cv2.line(user_frame, start_point, end_point, (0, 0, 255), 2)  # Красные линии

        # Отображаем совмещенное видео
        combined_frame = cv2.addWeighted(user_frame, 0.7, ideal_frame, 0.3, 0)  # Наложение кадров
        show_video_centered(combined_frame, 'Comparison')  # Отображаем по центру

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Используем задержку
            break

        frame_idx += 1

    ideal_cap.release()
    user_cap.release()
    cv2.destroyAllWindows()

# Функция для выбора файла
def select_file(label):
    file_path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    if file_path:
        label.config(text=file_path)

# Функция для просмотра идеального видео
def view_ideal_video():
    selected_video = ideal_video_var.get()
    if selected_video:
        ideal_video_path = IDEAL_VIDEOS[selected_video]
        resized_video_path = resize_video(ideal_video_path)  # Масштабируем видео
        cap = cv2.VideoCapture(resized_video_path)

        # Получаем FPS видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        delay = int(1000 / fps)  # Задержка между кадрами в миллисекундах

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Example', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):  # Используем задержку
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        messagebox.showwarning("Предупреждение", "Пожалуйста, выберите идеальное видео.")

# Функция для запуска сравнения
def start_comparison():
    ideal_video_name = ideal_video_var.get()
    user_video_path = user_video_label.cget("text")

    if not ideal_video_name:
        messagebox.showerror("Ошибка", "Пожалуйста, выберите идеальное видео.")
        return

    if not user_video_path:
        messagebox.showerror("Ошибка", "Пожалуйста, выберите пользовательское видео.")
        return

    try:
        # Создаем индикатор загрузки
        progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        progress.grid(row=5, column=0, columnspan=3, pady=10)
        progress["value"] = 0
        progress["maximum"] = 100

        def comparison_thread():
            try:
                # Получаем путь к идеальному видео
                ideal_video_path = IDEAL_VIDEOS[ideal_video_name]

                # Масштабирование видео
                progress["value"] = 10
                root.update_idletasks()
                ideal_resized_path = resize_video(ideal_video_path)  # Масштабируем видео
                progress["value"] = 20
                root.update_idletasks()
                user_resized_path = resize_video(user_video_path)  # Масштабируем видео
                progress["value"] = 30
                root.update_idletasks()

                # Обнаружение ключевых точек
                progress["value"] = 40
                root.update_idletasks()
                ideal_keypoints = detect_pose_in_video(ideal_resized_path)
                progress["value"] = 60
                root.update_idletasks()
                user_keypoints = detect_pose_in_video(user_resized_path)
                progress["value"] = 80
                root.update_idletasks()

                # Синхронизация кадров
                min_frames = min(len(ideal_keypoints), len(user_keypoints))
                ideal_keypoints = ideal_keypoints[:min_frames]
                user_keypoints = user_keypoints[:min_frames]

                # Сравнение движений
                similarity_scores = []
                for frame_idx in range(min_frames):
                    similarity = procrustes_analysis(user_keypoints[frame_idx], ideal_keypoints[frame_idx])
                    similarity_scores.append(similarity)

                # Вывод средней оценки точности
                max_distance = np.sqrt(2) * len(ideal_keypoints[0])  # Максимальное расстояние для нормализации
                avg_similarity = 100 * (1 - np.mean(similarity_scores) / max_distance)
                similarity = round(avg_similarity,0)
                messagebox.showinfo("penis", avg_similarity )
                if similarity < 70:
                    messagebox.showerror("Результат", f"Малое сходство. Возможные причины: \n 1. Вы неправильно выполняете упражнение \n 2. Неправильное позиционирование камеры \n 3. Неправильный темп выполнения упражнения")
                elif similarity > 70 and similarity < 95:
                    messagebox.showwarning("Результат", f"Отлично! Есть куда стремиться, но упражнение выполняется правильно!")
                elif similarity == 100:
                    messagebox.showinfo("Результат", "Вы идеально выполняете упражнение!")

                # Визуализация сравнения
                progress["value"] = 90
                root.update_idletasks()
                visualize_comparison(ideal_keypoints, user_keypoints, ideal_resized_path, user_resized_path)
                progress["value"] = 100
                root.update_idletasks()

            except Exception as e:
                messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            finally:
                progress.destroy()

        # Запускаем сравнение в отдельном потоке
        threading.Thread(target=comparison_thread, daemon=True).start()

    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

# Создание графического интерфейса
root = tk.Tk()
root.title("Коуч")

# Центрирование окна
window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Выбор идеального видео
tk.Label(root, text="Для записи Вашего видео, встаньте так, как указано на примере",font = ("Times new Roman",15)).grid(row = 0, column = 0, columnspan=3, padx = 10, pady = 1)
tk.Label(root, text="и начинайте запись сразу с выполнения упражнения",font = ("Times new Roman",15)).grid(row = 1, column = 0, columnspan=3, padx = 10, pady = 1)
tk.Label(root, text="Выберите упражнение:", font = ("Times new Roman",15)).grid(row=2, column=0, padx=10, pady=10)
ideal_video_var = tk.StringVar()
ideal_video_var.set("Упражнение 1")
ideal_video_dropdown = tk.OptionMenu(root, ideal_video_var, *IDEAL_VIDEOS.keys())
ideal_video_dropdown.grid(row=2, column=1, padx=10, pady=10)
tk.Button(root, text="Просмотр", command=view_ideal_video).grid(row=2, column=2, padx=10, pady=10)

# Выбор пользовательского видео
tk.Label(root, text="Выберите Ваше видео:",font = ("Times new Roman",15)).grid(row=3, column=0, padx=10, pady=10)
user_video_label = tk.Label(root, text="", fg="blue", wraplength=300)
user_video_label.grid(row=3, column=1, padx=10, pady=10)
tk.Button(root, text="Выбрать", command=lambda: select_file(user_video_label)).grid(row=3, column=2, padx=10, pady=10)

# Кнопка для запуска сравнения
tk.Button(root, text="Сравнить", command=start_comparison, bg="green", fg="white").grid(row=4, column=0, columnspan=3, pady=20)

# Запуск основного цикла
root.mainloop()
