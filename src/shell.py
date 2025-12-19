class Command:
	"""
	Класс команд, испольняемых в среде Shell
	"""
	def __init__(self, function, cmd_name: str, description: str = 'NO_DESCRIPTION'):
		self._func = function or (lambda context, args, kwargs: self.help_subcommands())
		self._name = cmd_name
		self._desc = description
		self._subcmds = {}

	def __call__(self, context, args: list, kwargs: dict):
		if len(args) >= 1:
			subcommand_name = args[0]
			return self._subcmds.get(subcommand_name, self._func)(
				context,
				args[(subcommand_name in self._subcmds.keys()):],
				kwargs)
		return self._func(context, args, kwargs)

	def stack(self, command):
		"""
		Делает из команды супер-команду. Добавляет подкоманды для нее.
		:param command: подкоманда
		:return: self
		"""
		self._subcmds[command.get_name()] = command
		return self

	def help_subcommands(self):
		"""
		Помощь по подкомандам.
		:return:
		"""
		for cmd_name, cmd in self._subcmds.items():
			print(f'\t{cmd_name}: {cmd.get_description()}')

	def get_name(self) -> str:
		"""
		Получет командлет команды.
		:return: str
		"""
		return self._name
	def get_description(self) -> str:
		"""
		Получает описание команды.
		:return: str
		"""
		return self._desc


HELP_DESC = 'Выводит список доступных команд и их описание.'
EXIT_DESC = 'Выход из программы.'

class Shell:
	"""
	Класс оболочки командной строки.
	"""
	def __init__(self):
		self._commands_dict = {}
		self._context = {
			'__shell__': self,
		}
		self._exit_code = None
		self.add_new_commands(
			Command(Shell._cmd_help, 'help', HELP_DESC),
			Command(Shell._cmd_exit, 'exit', EXIT_DESC)
		)

	@staticmethod
	def _cmd_help(context, args, kwargs):
		__shell__: Shell = context['__shell__']
		for cmd_name, cmd in __shell__._commands_dict.items():
			print(f'\t{cmd_name}: {cmd.get_description()}')
	@staticmethod
	def _cmd_exit(context, args, kwargs):
		__shell__: Shell = context['__shell__']
		__shell__._exit_code = 0
	@staticmethod
	def _cmd_unknown_command(context, args, kwargs):
		print("Unknown command.")

	def add_new_commands(self, *commands: Command):
		"""
		Добавляет команду в пул испольняемых команд средой.
		:param commands: множественные команды.
		:return:
		"""
		for _cmd in commands:
			self._commands_dict[_cmd.get_name()] = _cmd
	def execute_cmd(self, cmd_name: str, args, kwargs: dict):
		"""
		Выполняет команду по ее командлету.
		:param cmd_name: командлет.
		:param args: аргементы.
		:param kwargs: ключи и их значения.
		:return:
		"""
		self._commands_dict.get(cmd_name, Shell._cmd_unknown_command)(self._context, args, kwargs)

	@staticmethod
	def _parse_cmd_input(line):
		"""
		Парсит строчку команды.
		:param line:
		:return: кортеж (командлет, аргументы, ключи)
		"""
		cmd_name, *raw_args = line.split()
		is_key_list = list(map(lambda x: x.startswith('-'), raw_args))
		if True not in is_key_list:
			return cmd_name, raw_args, {}

		first_key_index = is_key_list.index(True)

		args = raw_args[:first_key_index]
		raw_kwargs = raw_args[first_key_index:]
		kwargs = {}
		
		i = 0
		while i < len(raw_kwargs):
			if raw_kwargs[i].startswith('-'):
				# если следующий аргумент есть и не начинается с '-', значит это значение
				if i + 1 < len(raw_kwargs) and not raw_kwargs[i + 1].startswith('-'):
					kwargs[raw_kwargs[i]] = raw_kwargs[i + 1]
					i += 2
				else:
					kwargs[raw_kwargs[i]] = None
					i += 1
			else:
				i += 1  # пропускаем неключевые слова (например, model, train)
		return cmd_name, args, kwargs

	def enter_message_loop(self):
		"""
		Заходит в цикл сообщений.
		:return:
		"""
		while self._exit_code is None:
			print(">> ", end='')
			__input: str = input()
			if not __input:
				continue
			cmd_name, args, kwargs = Shell._parse_cmd_input(__input)

			try:
				self.execute_cmd(cmd_name, args, kwargs)
			except Exception as ex:
				print(ex)