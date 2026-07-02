# 02 Preprocessing / Document Building

Ответственность:

- очистка текстов model cards;
- выделение title, description, tags, task, license и других полезных полей;
- сборка поисковых документов;
- дедупликация и фильтрация плохих документов;
- преобразование `ModelCard` в `SearchDocument`.

Ожидаемый выход модуля: коллекция `SearchDocument`.
