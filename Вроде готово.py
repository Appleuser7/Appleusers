import os
import librosa
import numpy as np
from tensorflow.keras import layers, models


# Функция для вычисления MFCC для одного аудиофайла
def compute_mfcc(audio_file, num_mfcc=20):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    return mfcc


# Функция для подготовки данных MFCC для одной базы данных
def prepare_data(database_path, num_mfcc=20):
    mfcc_data = []
    audio_files = librosa.util.find_files(database_path)
    for audio_file in audio_files:
        mfcc = compute_mfcc(audio_file, num_mfcc)
        mfcc_data.append(mfcc)
    return mfcc_data


# Функция для вычисления среднего значения MFCC для каждой базы данных
def mean_mfcc(database):
    mean_values = []
    for mfcc_data in database:
        mean_mfcc = np.mean(mfcc_data, axis=1)
        mean_values.append(mean_mfcc)
    return mean_values


# Пути к папкам с аудиофайлами баз данных
database1_path = 'D:\\ai'
database2_path = 'D:\\real'

# Получение данных MFCC для каждой базы данных
mfcc_data1 = prepare_data(database1_path)
mfcc_data2 = prepare_data(database2_path)

# Вычисление средних значений MFCC для каждой базы данных
mean_values1 = mean_mfcc(mfcc_data1)
mean_values2 = mean_mfcc(mfcc_data2)

# Сравнение средних значений MFCC
min_len = min(len(mean_values1), len(mean_values2))
accuracies = []  # Список для хранения точностей всех проверок

for i in range(min_len):
    if i < len(mean_values1) and i < len(mean_values2):
        mse = np.mean(
            (mean_values1[i] - mean_values2[i]) ** 2)  # Среднеквадратичное отклонение (MSE) между MFCC двух баз данных
        print(f"Среднеквадратичное отклонение для аудиофайла {i + 1}: {mse}")

        # Определение категории аудиофайла
        if 'ai' in os.path.basename(database1_path):
            category1 = 'сгенерирован голос'
            category2 = 'человеческий голос'
        else:
            category1 = 'человеческий голос'
            category2 = 'сгенерирован голос'

        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(20,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')  # Два выходных нейрона для двух классов
        ])

        # Компиляция модели
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',  # Использование categorical_crossentropy
                      metrics=['accuracy'])

        # Подготовка данных
        X_train = []
        y_train = []

        # Добавление данных из первой базы данных (сгенерированный голос)
        for mse_value in mean_values1:
            X_train.append(mse_value)
            y_train.append([0, 1])  # 0 для человеческого голоса, 1 для сгенерированного голоса

        # Добавление данных из второй базы данных (человеческий голос)
        for mse_value in mean_values2:
            X_train.append(mse_value)
            y_train.append([1, 0])  # 1 для сгенерированного голоса, 0 для человеческого голоса

        # Преобразование в массивы numpy
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Пути к папке с аудиофайлами для проверки
        test_database_path = 'C:\\Users\\kiril\\OneDrive\\Рабочий стол\\test'

        # Определение типа голоса (сгенерированный или человеческий)
        is_generated_voice = 'generated' in test_database_path.lower()

        # Получение данных MFCC для проверки
        mfcc_test_data = prepare_data(test_database_path)

        # Вычисление средних значений MFCC для проверки
        mean_test_values = mean_mfcc(mfcc_test_data)

        # Создание X_test и y_test для проверки
        X_test = []
        y_test = []

        # Добавление данных для проверки и соответствующих меток
        for mse_value in mean_test_values:
            X_test.append(mse_value)
            if is_generated_voice:
                y_test.append([0, 1])  # 0 для человеческого голоса, 1 для сгенерированного голоса
            else:
                y_test.append([1, 0])  # 1 для сгенерированного голоса, 0 для человеческого голоса

        # Преобразование в массивы numpy
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Обучение модели
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        # Оценка эффективности модели на тестовых данных
        _, accuracy = model.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    # Вывод конечного результата
    average_accuracy = np.mean(accuracies) * 100
    print("Средняя точность всех проверок: {:.2f}%".format(average_accuracy))
