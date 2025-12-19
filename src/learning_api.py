import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.ops.gen_batch_ops import batch

# Shortcuts
keras = tf.keras
layers = keras.layers

def check_tf_gpus():
	"""
	Поиск устройств для вычисления.
	"""
	print("Версия TensorFlow:", tf.__version__)
	gpus = tf.config.list_physical_devices('GPU')
	cpus = tf.config.list_physical_devices('CPU')
	print("Доступные GPU:", gpus)
	print("Доступные CPU:", cpus)
	if gpus:
		print("GPU доступен!")
		for gpu in gpus:
			print(f"GPU: {gpu}")
	else:
		print("GPU не доступен, используется CPU")
def check_cuda_versions():
	"""
	Провека версий CUDA. (Windows-based)
	"""
	cuda_path = os.path.join('C:', 'Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA')
	if os.path.exists(cuda_path):
		versions = [d for d in os.listdir(cuda_path) if d.startswith('v')]
		print(f"Найдены версии CUDA: {versions}")

		# Используем 12.6
		cuda_12_6_path = os.path.join(cuda_path, "v12.6")
		if os.path.exists(cuda_12_6_path):
			print("Используется CUDA 12.6")
			return cuda_12_6_path
		else:
			print("CUDA 12.6 не найдена")
	return None
def check_environment():
	"""
	Проверка переменных окружения.
	"""
	print("=== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===")
	path = os.environ.get('PATH', '')
	# Проверка CUDA в PATH
	cuda_versions = ['v12.6', 'v12.5']
	found = False

	for version in cuda_versions:
		if os.path.join('CUDA', version, 'bin') in path:
			print(f"CUDA {version} в PATH")
			found = True
	if not found:
		print("CUDA не найдена в PATH")

def load_mnist_image(path: str) -> np.array:
	"""
	Переводит изображение по пути в np.array (тензор), также нормирует его.
	:param path: путь до изображения
	:return: тензор
	"""
	img = Image.open(path).convert("L").resize((28, 28))
	arr = np.array(img, dtype=np.float32)
	arr = arr / 255.0                # нормализация
	arr = np.expand_dims(arr, -1)    # канал: (28, 28) -> (28, 28, 1)
	return arr

def cmd_gpu_check(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Показывает информацию об используемых вычислительных устройствах,
	переменных среды и проверяет версию CUDA.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	check_tf_gpus(), check_cuda_versions(), check_environment()

def cmd_model_new(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Создавет модель из файла по пути.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	LAYERS_MAP = {
		'Input': layers.Input,
		'Conv2D': layers.Conv2D,
		'MaxPooling2D': layers.MaxPooling2D,
		'Flatten': layers.Flatten,
		'Dropout': layers.Dropout,
		'Dense': layers.Dense,
	}
	file_path = args[0]

	with (open(file_path, 'r') as file):
		model = keras.Sequential()

		for line in file:
			line = line.strip()
			if not line:
				continue

			layer_type = line.split('(')[0]
			if layer_type not in LAYERS_MAP:
				raise KeyError(f'Unknown layer "{layer_type}".')

			layer = eval(
				line,  {'__builtins__': {}}, LAYERS_MAP
			)
			model.add(layer)

		model.compile(
			optimizer=keras.optimizers.Adam(learning_rate=1e-3),
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'],
		)

		context['model'] = model

def cmd_model_load(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Загружает модель из ее дампа.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	file_path = args[0]
	model = keras.models.load_model(file_path)
	context['model'] = model

def cmd_model_info(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Выводит инфорацию о моделе.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	model = context['model']
	model.summary()

def cmd_model_train(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Обучает модель.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	model = context['model']

	epochs = int(args[0])
	batch_size = 128 if '-b' not in kwargs else int(kwargs['-b'])
	val_size = 5000 if '-v' not in kwargs else int(kwargs['-v'])

	print(f'{epochs=}', f'{batch_size=}', f'{val_size=}')

	# 1) Загрузка и подготовка данных MNIST
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	# Нормализация к [0,1]
	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0
	# Добавляем канал (28, 28) -> (28, 28, 1)
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)

	# Выделим валидационный набор из трейна
	x_val, y_val = x_train[-val_size:], y_train[-val_size:]
	x_train, y_train = x_train[:-val_size], y_train[:-val_size]

	# 2) tf.data пайплайн для скорости
	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
		.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
		.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
		.batch(batch_size).prefetch(tf.data.AUTOTUNE)

	# Коллбеки: ранняя остановка и чекпоинт
	callbacks = [
		keras.callbacks.ModelCheckpoint('mnist_cnn.best.keras', monitor='val_accuracy', save_best_only=True),
	]

	if '-p' in kwargs and bool(kwargs['-p']):
		patience = int(kwargs['-p'])
		callbacks.append(keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True))

	if '-l' in kwargs and kwargs['-l'] is None:
		callbacks.append(keras.callbacks.TensorBoard(log_dir='logs'))

	# 4) Обучение
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
		callbacks=callbacks,
	)
	if '-h' in kwargs and kwargs['-h'] is None:
		context['history'] = history

	# 5) Оценка на тесте
	test_loss, test_acc = model.evaluate(test_ds)
	print(f"Тестовая точность: {test_acc:.4f}, тестовый лосс: {test_loss:.4f}")

def cmd_model_predict(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Использует модель для прогнозирования по изображению.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	model = context['model']
	file_paths = args

	batch = np.stack([load_mnist_image(path) for path in file_paths], axis=0)
	probs = model.predict(batch, verbose=0)
	preds = np.argmax(probs, axis=-1)

	for p, cls, prob in zip(file_paths, preds, probs[np.arange(len(file_paths)), preds]):
		print(f"{p}: pred={cls}, p={prob:.3f}")

def cmd_model_save(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Сохраняет дамп модели на диск.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	model = context['model']
	save_path = args[0]
	model.save(save_path)
	print(f"Модель сохранена в: {os.path.abspath(save_path)}")

def cmd_plot(context, args, kwargs):
	"""
	API-функция, предназначенная для Shell. Строит график зависимостей loss от epoch.
	:param context: контекст использования.
	:param args: аргументы.
	:param kwargs: ключи и их значения.
	"""
	history = context['history']
	Es = history.epoch

	plt.figure(figsize=(6, 4))
	plt.plot(Es, history.history['loss'], marker='o', label='loss')
	plt.plot(Es, history.history['val_loss'], marker='o', label='val_loss')
	plt.title('Зависимость ошибки от числа эпох.')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.show()