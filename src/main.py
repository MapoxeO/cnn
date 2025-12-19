from shell import Command, Shell
import learning_api as learning

GPU_CHECK_DESC = 'Выводит информацию о том, какие устройства для вычислений доступны, а также версия CUDA'
MODEL_DESC = 'Супер-команда, работает с моделью ИНС.'
MODEL_NEW_DESC = 'Строит модель по файлу.'
MODEL_LOAD_DESC = 'Загружает модель из дампа keras-файла.'
MODEL_INFO_DESC = 'Выводит информацию о структуре и количестве параметров модели.'
MODEL_TRAIN_DESC = 'Тренирует модель на датасете MNIST.'
MODEL_PREDICT_DESC = 'Делает предсказания моделью по фотографии.'
MODEL_SAVE_DESC = 'Сохраняет дамп модели на диск.'
PLOT_DESC = 'Выводит график зависимостей ошибок от числа эпох. '

def main():
	shell: Shell = Shell()
	shell.add_new_commands(
		Command(learning.cmd_gpu_check, GPU_CHECK_DESC),
		Command(None, 'model', MODEL_DESC) \
			.stack(Command(learning.cmd_model_new, 'new', MODEL_NEW_DESC)) \
			.stack(Command(learning.cmd_model_load, 'load', MODEL_LOAD_DESC)) \
			.stack(Command(learning.cmd_model_info, 'info', MODEL_INFO_DESC)) \
			.stack(Command(learning.cmd_model_train, 'train', MODEL_TRAIN_DESC)) \
			.stack(Command(learning.cmd_model_predict, 'predict', MODEL_PREDICT_DESC)) \
			.stack(Command(learning.cmd_model_save, 'save', MODEL_SAVE_DESC)),
		Command(learning.cmd_plot, 'plot', PLOT_DESC),
	)
	shell.enter_message_loop()

if __name__ == '__main__':
	main()
